"""A-bench adapters: run each released checkpoint with its authors' own loader and model code.

make(model, ds, data_root) returns run(K, pin) -> iterator of (obs (N,8,2), gt (N,12,2), samples (K,N,12,2),
mu (N,12,2) | None), absolute world coordinates, torch tensors on the CPU. data_root is the folder that holds
<ds>/test/*.txt (a re-indexed copy for the perturbed conditions). pin=True (models in NEEDS_PIN) replaces the released
index-based noise by noise keyed on SEED and each pedestrian's own observed track (protocol amendment 1). Each job runs
in its own process, so repository modules never collide.
"""
import os, sys, glob, pickle, hashlib, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
import networkx as nx

REPOS = os.environ.get('BENCH_REPOS', '/content/repos')
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{'weights_only': False, **k})  # releases pickle whole objects (torch >= 2.6)
for _n, _t in (('float', float), ('int', int), ('bool', bool), ('object', object)):  # numpy >= 1.24 removed aliases
    if _n not in np.__dict__:
        setattr(np, _n, _t)
if not hasattr(nx, 'from_numpy_matrix'):                                 # networkx >= 3 renamed it
    nx.from_numpy_matrix = nx.from_numpy_array
if not torch.cuda.is_available():                                          # Kaggle CPU sessions: repository .cuda() -> no-op
    # shortcut: covers .cuda() only; code passing device='cuda' (Graph-TERN, NPSN SGCN, PECNet) still needs a GPU
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self

if os.environ.get('BENCH_VARIANT') == 'F':                               # B-retrain: pedestrian-wise kernels
    import pedwise
    pedwise.apply()
CKPT = os.environ.get('BENCH_CKPT') and os.path.join(os.environ['BENCH_CKPT'], '')  # B-retrain: a trained checkpoint dir

SEED, KEYS, CALL = 0, [], [0]                                             # pinned noise: job seed, keys of the current batch
NEEDS_PIN = {'PECNet', 'GraphTERN', 'Singular', 'Implicit'}


def okeys(obs):
    """Unsigned 63-bit key of each pedestrian's observed track (N,8,2) in mm: its noise seed when pinned."""
    mm = np.round(obs.detach().double().cpu().numpy().reshape(len(obs), -1) * 1000).astype(np.int64)
    return [int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), 'little') >> 1 for r in mm]


def keyed(*extra):
    return [np.random.default_rng([SEED, k, *extra]) for k in KEYS]


def enter(repo):
    path = os.path.join(REPOS, repo)
    os.chdir(path)
    sys.path.insert(0, path)
    return path


def cpu(*ts):
    return [None if t is None else t.detach().float().cpu() for t in ts]


def gauss(V):
    """(..., 5) bivariate-Gaussian head -> mean, covariance (Social-STGCNN convention)."""
    sx, sy, r = V[..., 2].exp(), V[..., 3].exp(), V[..., 4].tanh()
    cov = torch.stack([sx * sx, r * sx * sy, r * sx * sy, sy * sy], -1).view(*V.shape[:-1], 2, 2)
    return V[..., :2], cov


def sgan_out(obs_traj, gt_traj, V, K):
    """obs_traj (1,N,2,8), gt_traj (1,N,2,12), V (12,N,5) relative steps -> absolute outputs."""
    obs, gt = obs_traj[0].permute(0, 2, 1), gt_traj[0].permute(0, 2, 1)
    last = obs[:, -1]
    mu, cov = gauss(V)
    s = MultivariateNormal(mu, cov).sample((K,))                      # K,12,N,2
    return cpu(obs, gt, (s.cumsum(1) + last).permute(0, 2, 1, 3), (mu.cumsum(0) + last).permute(1, 0, 2))


def stgcnn(ds, root):
    enter('Social-STGCNN')
    import utils, model as M
    d = CKPT or f'checkpoint/social-stgcnn-{ds}/'
    a = pickle.load(open(d + 'args.pkl', 'rb'))
    net = M.social_stgcnn(n_stgcnn=a.n_stgcnn, n_txpcnn=a.n_txpcnn, output_feat=a.output_size, seq_len=a.obs_seq_len,
                          kernel_size=a.kernel_size, pred_seq_len=a.pred_seq_len).cuda()
    net.load_state_dict(torch.load(d + 'val_best.pth')); net.eval()
    dset = utils.TrajectoryDataset(f'{root}/{ds}/test/', obs_len=8, pred_len=12, skip=1, norm_lap_matr=True)

    def run(K, pin=False):
        for b in DataLoader(dset, batch_size=1):
            b = [t.cuda() for t in b]
            V, _ = net(b[6].permute(0, 3, 1, 2), b[7].squeeze())
            yield sgan_out(b[0], b[1], V.permute(0, 2, 3, 1)[0], K)
    return run


def dmrgcn(ds, root):
    enter('DMRGCN')
    from utils import TrajectoryDataset
    from model import social_dmrgcn
    d = CKPT or f'checkpoints/social-dmrgcn-{ds}-experiment_tp4_de80/'
    a = pickle.load(open(d + 'args.pkl', 'rb'))
    net = social_dmrgcn(n_stgcn=a.n_stgcn, n_tpcnn=a.n_tpcnn, output_feat=a.output_size, kernel_size=a.kernel_size,
                        seq_len=a.obs_seq_len, pred_seq_len=a.pred_seq_len).cuda()
    net.load_state_dict(torch.load(d + f'{ds}_best.pth')); net.eval()
    dset = TrajectoryDataset(f'{root}/{ds}/test/', obs_len=a.obs_seq_len, pred_len=a.pred_seq_len, skip=1)

    def run(K, pin=False):
        for b in DataLoader(dset, batch_size=1):
            b = [t.cuda() for t in b]
            V, _ = net(b[-4].permute(0, 3, 1, 2), b[-3])
            yield sgan_out(b[0], b[1], V.permute(0, 2, 3, 1)[0], K)
    return run


class KeyedMixture(torch.distributions.MixtureSameFamily):
    """Graph-TERN's endpoint GMM (batch (1, N), M components, 2-D) sampled with each pedestrian's own keyed noise."""

    def sample(self, sample_shape=torch.Size()):
        n = sample_shape[0]
        probs = self.mixture_distribution.probs.detach()
        loc, scale = self.component_distribution.base_dist.loc, self.component_distribution.base_dist.scale
        B, N, M, D = loc.shape
        assert B == 1 and N == len(KEYS), (loc.shape, len(KEYS))
        CALL[0] += 1
        out = torch.empty(n, B, N, D, device=loc.device, dtype=loc.dtype)
        cdf = probs[0].cumsum(-1)
        for i, r in enumerate(keyed(CALL[0])):
            c = torch.searchsorted(cdf[i].contiguous(), torch.from_numpy(r.random(n)).to(cdf)).clamp(max=M - 1)
            out[:, 0, i] = loc[0, i, c] + scale[0, i, c] * torch.from_numpy(r.standard_normal((n, D))).to(loc)
        return out


def graphtern(ds, root):
    enter('GraphTERN')
    from utils.dataloader import TrajectoryDataset
    from graphtern.model import graph_tern
    d = CKPT or f'checkpoint/graph-tern_{ds}_experiment/'
    a = pickle.load(open(d + 'args.pkl', 'rb'))
    net = graph_tern(n_epgcn=a.n_epgcn, n_epcnn=a.n_epcnn, n_trgcn=a.n_trgcn, n_trcnn=a.n_trcnn, seq_len=a.obs_seq_len,
                     pred_seq_len=a.pred_seq_len, n_ways=a.n_ways, n_smpl=a.n_smpl).cuda()
    net.load_state_dict(torch.load(d + f'{ds}_best.pth'), strict=False); net.eval()
    dset = TrajectoryDataset(f'{root}/{ds}/test/', obs_len=a.obs_seq_len, pred_len=a.pred_seq_len, skip=1)

    import graphtern.model as GM
    released = GM.MixtureSameFamily

    def run(K, pin=False):
        net.n_smpl = K
        GM.MixtureSameFamily = KeyedMixture if pin else released
        for b in DataLoader(dset, batch_size=1):
            S_obs, S_trgt = [t.cuda() for t in b[-2:]]
            KEYS[:], CALL[0] = okeys(S_obs[0, 0].permute(1, 0, 2)), 0
            if pin:
                random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            _, _, V_refi, _ = net(S_obs, pruning=4, clustering=True)  # K,12,N,2 absolute
            yield cpu(S_obs[0, 0].permute(1, 0, 2), S_trgt[0, 0].permute(1, 0, 2), V_refi.permute(0, 2, 1, 3), None)
    return run


def implicit(ds, root):
    enter('Social-Implicit')
    import utils
    from model import SocialImplicit
    from CFG import CFG
    net = SocialImplicit(spatial_input=CFG['spatial_input'], spatial_output=CFG['spatial_output'],
                         temporal_input=CFG['temporal_input'], temporal_output=CFG['temporal_output'], bins=CFG['bins'],
                         noise_weight=CFG['noise_weight_eth' if ds == 'eth' else 'noise_weight']).cuda()
    net.load_state_dict(torch.load(f'checkpoint/social-implicit-{ds}/val_best.pth'))
    net.cuda().double().eval()
    dset = utils.TrajectoryDataset(f'{root}/{ds}/test/', obs_len=8, pred_len=12, skip=1, norm_lap_matr=True)

    def run(K, pin=False):
        for b in DataLoader(dset, batch_size=1):
            b = [t.cuda().double() for t in b]
            obs, gt = b[0][0].permute(0, 2, 1), b[1][0].permute(0, 2, 1)
            if pin:
                torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            V = net(b[6].permute(0, 3, 1, 2), b[0], KSTEPS=K)            # K,2,12,N relative steps
            s = V.permute(0, 2, 3, 1).cumsum(1) + obs[:, -1]             # K,12,N,2
            yield cpu(obs, gt, s.permute(0, 2, 1, 3), None)
    return run


def npsn_sgcn(ds, root):
    enter('NPSN')
    from baselines.sgcn import get_dataloader, get_model
    from baselines.converter import get_sgcn_identity
    net = get_model().cuda()
    net.load_state_dict(torch.load(f'pretrained/sgcn/{ds}/val_best.pth')); net.eval()
    loader, _ = get_dataloader(f'{root}/{ds}/', 'test', 8, 12, 1)

    def run(K, pin=False):
        for b in loader:
            obs_traj, gt_traj = b[0].cuda(), b[1].cuda()
            V_obs = b[-2].cuda()
            V = net(V_obs, get_sgcn_identity(V_obs.shape)).squeeze(0)   # 12,N,5
            assert V.shape[-1] == 5 and V.shape[0] == 12, V.shape
            yield sgan_out(obs_traj, gt_traj, V, K)
    return run


def npsn_pecnet(ds, root):
    enter('NPSN')
    from baselines.pecnet import get_dataloader, get_model, model_forward_pre_hook
    net = get_model().cuda()                                             # also sets the module's data_scale
    net.load_state_dict(torch.load(f'pretrained/pecnet/{ds}/val_best.pth')); net.eval()
    loader, _ = get_dataloader(f'{root}/{ds}/', 'test', 8, 12, 1)      # 1 scene per yield for scene-level locality
    dev = torch.device('cuda')

    def run(K, pin=False):
        for b in loader:
            obs_traj, gt_traj, mask, x, _, init, scale = model_forward_pre_hook(b)
            obs, gt = obs_traj.permute(0, 2, 1), gt_traj.permute(0, 2, 1)
            if pin:                                                       # latent z = noise * sigma, per pedestrian
                KEYS[:] = okeys(obs)
                Z = torch.from_numpy(np.stack([r.standard_normal((K, net.zdim)) for r in keyed()], 1)).float().cuda()
            out = []
            for k in range(K):
                dest = net.forward(x, init, device=dev, noise=Z[k] if pin else None)
                fut = net.predict(x, dest, mask, init)
                out.append(torch.cat([fut, dest], 1).view(-1, 12, 2) / scale + obs[:, :1])
            yield cpu(obs, gt, torch.stack(out), None)
    return run


ET_TAGS = {'stgcnn': 'STGCNN', 'sgcn': 'SGCN', 'dmrgcn': 'DMRGCN', 'gpgraphstgcnn': 'GPGraph-STGCNN',
           'gpgraphsgcn': 'GPGraph-SGCN', 'graphtern': 'GraphTERN', 'implicit': 'Implicit', 'pecnet': 'PECNet',
           'agentformer': 'AgentFormer', 'lbebm': 'LBEBM'}


class KeyedNumpy:
    """numpy stand-in for SingularTrajectory's sampler: default_rng(seed=0).normal(size=(N, ...)) gives row i the
    noise keyed on pedestrian i instead of the noise of index i."""

    def __getattr__(self, name):
        return getattr(np, name)

    class random:
        @staticmethod
        def default_rng(seed=None):
            return KeyedNumpy._Rows()

    class _Rows:
        def normal(self, loc=0.0, scale=1.0, size=None):
            assert size[0] == len(KEYS), (size, len(KEYS))
            return np.stack([r.normal(loc, scale, size[1:]) for r in keyed()])


def eigen_family(repo, cfg, tag, root):
    """EigenTrajectory / SingularTrajectory: build the trainer as trainval.py --test does, run its own test loop and
    capture the model inputs and outputs."""
    enter(repo)
    import argparse
    import baseline
    import utils
    import utils.trainer as T
    Model = getattr(__import__(repo), repo)
    hp = utils.get_exp_config(cfg)
    hp.dataset_dir = root + '/'
    full = T.get_dataloader
    T.get_dataloader = lambda d, phase, *a, **k: full(d, phase, *a, **k) if phase == 'test' else None
    names = [s for s in T.__dict__.keys() if hp.baseline in s.lower()]
    names = names or [s for s in T.__dict__.keys() if s.lower().endswith('trainer') and hp.baseline[:7] in s.lower()]
    b = getattr(baseline, hp.baseline)
    hook = utils.DotDict({'model_forward_pre_hook': b.model_forward_pre_hook, 'model_forward': b.model_forward,
                          'model_forward_post_hook': b.model_forward_post_hook})
    args = argparse.Namespace(cfg=cfg, tag=tag, gpu_id='0', test=True)
    tr = getattr(T, names[0])(base_model=b.TrajectoryPredictor, model=Model, hook_func=hook, args=args, hyper_params=hp)
    tr.load_model()
    cap = []
    fwd = tr.model.forward

    def forward(obs, *a, **k):
        cap.append(obs)
        KEYS[:] = okeys(obs)
        return fwd(obs, *a, **k)
    tr.model.forward = forward
    diff = sys.modules.get('baseline.transformerdiffusion.model')        # SingularTrajectory's sampler module
    released_np = diff.np if diff else None

    def grab(recon, gt):
        cap.append((recon, gt))
        return np.zeros(gt.shape[-3])

    def reset_metric():
        tr.stats_func, tr.stats_meter = {'C': grab}, {'C': utils.AverageMeter()}
    tr.reset_metric = reset_metric

    def run(K, pin=False):
        assert K == hp.num_samples, (K, hp.num_samples)
        assert not pin or diff, 'pinning is only defined for the SingularTrajectory sampler'
        if diff:
            diff.np = KeyedNumpy() if pin else released_np
        cap.clear()
        with torch.no_grad():
            tr.test()
        for i in range(0, len(cap), 2):
            obs, (recon, gt) = cap[i], cap[i + 1]
            yield cpu(obs, gt.reshape(-1, 12, 2), recon, None)
    return run


def eigen(base):
    return lambda ds, root: eigen_family('EigenTrajectory', f'config/eigentrajectory-{base}-{ds}.json',
                                         f'EigenTrajectory-{ET_TAGS[base]}', root)


def singular(ds, root):
    return eigen_family('SingularTrajectory', f'config/stochastic/singulartrajectory-transformerdiffusion-{ds}.json',
                        'SingularTrajectory-stochastic', root)


def eqmotion(ds, root):
    """EqMotion ('diverse' release): its own ego-centric preprocessing (process_eth_data_diverse.py, test part) is
    replicated on the raw files under root/<ds>/ so that re-indexed copies can be evaluated."""
    path = enter('EqMotion')
    from eth_ucy.model_t import EqMotion
    os.chdir(os.path.join(path, 'eth_ucy'))
    sys.path.insert(0, os.getcwd())
    from dataloader import data_generator
    total_num = {'univ': 2, 'zara1': 3}.get(ds, 1)
    scale = 1.6 if ds == 'eth' else 1.0
    os.chdir(os.path.dirname(root))                                      # root = <x>/eth_ucy/data; it reads ../eth_ucy/data
    gen = data_generator(ds, 8, 12, 1, split='test', phase='testing')
    past, fut, nums, sizes = [], [], [], []
    while not gen.is_epoch_end():
        data = gen()
        if data is None:
            continue
        loc, loc_end = np.array(torch.stack(data['pre_motion_3D'])), np.array(torch.stack(data['fut_motion_3D']))
        n = loc.shape[0]
        sizes.append(n)
        for i in range(n):
            dist = np.linalg.norm(loc[:, -1] - loc[i:i + 1, -1], axis=-1)
            if n < total_num:                                             # ego first, then the others in file order
                idx = [i] + [j for j in range(n) if j != i]
            else:                                                         # nearest within 5 m, ego first
                idx = np.argsort(dist)[:min(int((dist < 5).sum()), total_num)]
            p, f = np.zeros((total_num, 8, 2)), np.zeros((total_num, 12, 2))
            p[:len(idx)], f[:len(idx)] = loc[idx], loc_end[idx]
            past.append(p); fut.append(f); nums.append(len(idx))
    past, fut = torch.tensor(np.array(past), dtype=torch.float) / scale, torch.tensor(np.array(fut), dtype=torch.float)
    nums = torch.tensor(nums)
    net = EqMotion(in_node_nf=8, in_edge_nf=2, hidden_nf=64, in_channel=8, hid_channel=128 if ds == 'zara1' else 64,
                   out_channel=12, device=torch.device('cuda'), n_layers=4, recurrent=True, norm_diff=False, tanh=False)
    net.load_state_dict(torch.load(os.path.join(path, f'eth_ucy/saved_models/{ds}_ckpt.pth.tar'))['state_dict'], strict=False)
    net.eval()

    def run(K, pin=False):
        preds = []
        for s in range(0, len(past), 100):
            loc, num = past[s:s + 100].cuda(), nums[s:s + 100].cuda().int()
            vel = torch.zeros_like(loc)
            vel[:, :, 1:] = loc[:, :, 1:] - loc[:, :, :-1]
            vel[:, :, 0] = vel[:, :, 1]
            pred, _ = net(vel.norm(dim=-1), loc, vel, num)               # B,N,20,12,2
            assert pred.shape[2] == K, pred.shape
            preds.append((pred[:, 0].permute(1, 0, 2, 3) * scale).detach().float().cpu())
        preds = torch.cat(preds, 1) if preds else torch.zeros(K, 0, 12, 2)
        p = 0
        for n in sizes:
            yield (past[p:p + n, 0] * scale).float(), fut[p:p + n, 0].float(), preds[:, p:p + n], None
            p += n
    return run


MODELS = {'STGCNN': ('Social-STGCNN', stgcnn), 'Implicit': ('Social-Implicit', implicit), 'SGCN': ('NPSN', npsn_sgcn),
          'DMRGCN': ('DMRGCN', dmrgcn), 'GraphTERN': ('GraphTERN', graphtern), 'PECNet': ('NPSN', npsn_pecnet),
          'EqMotion': ('EqMotion', eqmotion), 'Singular': ('SingularTrajectory', singular)}
MODELS.update({f'ET-{v}': ('EigenTrajectory', eigen(k)) for k, v in ET_TAGS.items()})
CLASS_D = ['STGCNN', 'Implicit', 'SGCN', 'DMRGCN', 'GraphTERN', 'ET-STGCNN', 'ET-SGCN', 'ET-DMRGCN',
           'ET-GPGraph-STGCNN', 'ET-GPGraph-SGCN', 'ET-GraphTERN', 'ET-Implicit']
CLASS_I = ['PECNet', 'EqMotion', 'ET-PECNet', 'ET-AgentFormer', 'ET-LBEBM', 'Singular']
assert sorted(CLASS_D + CLASS_I) == sorted(MODELS)
