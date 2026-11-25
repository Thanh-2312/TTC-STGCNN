"""Shared research helpers: cached datasets for each graph type and a fast vectorized evaluator."""
import os
import sys
import math
import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import social_stgcnn  # noqa: E402
from utils import TrajectoryDataset, seq_to_graph_hybrid  # noqa: E402
from metrics import (detect_group_pairs, col1_scene, col1_check_scene,  # noqa: E402
                     col2_rate_scene, col2_check_scene, avg_energy_scene,
                     min_dist_pairs_pred_pred)

DATA_ROOT = os.environ.get('TTC_DATA_ROOT', os.path.join(os.path.dirname(ROOT), 'datasets_eth_hotels_'))
SDD_ROOT = os.environ.get('TTC_SDD_ROOT', os.path.join(ROOT, 'data', 'sdd_eth_format'))  # output of src/sdd_prep.py
CACHE = os.environ.get('TTC_CACHE', os.path.join(ROOT, 'data', 'cache'))
DATASETS = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
DERIVED = ('A_eye', 'A_hyb')
LEAN = {'sdd'}  # large scenes: cache trajectories only; graphs are computed per batch (fast.batch_graphs)


def graph_official(seq_rel):
    """Social-STGCNN (Mohamed et al. 2020) graph: 1/||v_i - v_j|| on relative coords, then I - D^-1/2 A D^-1/2.
    Note: utils.seq_to_graph uses ||.|| (not its inverse), so it is NOT the published baseline."""
    seq_rel = seq_rel.numpy()
    N, _, T = seq_rel.shape
    A = np.zeros((T, N, N))
    for s in range(T):
        step = seq_rel[:, :, s]
        d = np.linalg.norm(step[:, None] - step[None], axis=-1)
        with np.errstate(divide='ignore'):
            a = np.where(d > 0, 1.0 / d, 0.0)
        np.fill_diagonal(a, 1.0)  # official code sets A_hh = 1 before nx.normalized_laplacian_matrix
        dinv = 1.0 / np.sqrt(a.sum(1))
        A[s] = np.eye(N) - dinv[:, None] * a * dinv[None]
    return torch.from_numpy(A).float()


def graph_hybrid(seq_rel, E, c=1.0):
    """Official 1/||dv|| affinity + c*tanh(E_ttc) (collision urgency), same normalized Laplacian as graph_official."""
    seq_rel = seq_rel.numpy()
    N, _, T = seq_rel.shape
    A = np.zeros((T, N, N))
    for s in range(T):
        step = seq_rel[:, :, s]
        d = np.linalg.norm(step[:, None] - step[None], axis=-1)
        with np.errstate(divide='ignore'):
            a = np.where(d > 0, 1.0 / d, 0.0) + c * np.tanh(E[s].numpy())
        np.fill_diagonal(a, 1.0)
        dinv = 1.0 / np.sqrt(a.sum(1))
        A[s] = np.eye(N) - dinv[:, None] * a * dinv[None]
    return torch.from_numpy(A).float()


def _add_graphs(scenes, need=DERIVED):
    """Graphs derived on load (not cached): A_eye = no interaction, A_hyb = official + TTC urgency."""
    if scenes and 'A_off' not in scenes[0]:  # lean cache: graphs come from scene_graph / batch_graphs
        return scenes
    for s in scenes:
        T, N = s['A_off'].shape[:2]
        if 'A_eye' in need:
            s['A_eye'] = torch.eye(N).expand(T, N, N).contiguous()
        if 'A_hyb' in need:
            s['A_hyb'] = graph_hybrid(s['obs_rel'], s['E_obs'])
    return scenes


def load_split(ds, split, obs_len=8, pred_len=12, need=DERIVED):
    """Returns dict of tensors per scene with A for every graph type (cached); `need` limits the derived graphs."""
    path = os.path.join(CACHE, f'{ds}_{split}.pt')
    if os.path.exists(path):
        return _add_graphs(torch.load(path, weights_only=False), need)
    root = SDD_ROOT if ds == 'sdd' else DATA_ROOT
    if ds in LEAN:
        return _lean_split(os.path.join(root, ds, split) + '/', path, obs_len, pred_len)
    d = TrajectoryDataset(os.path.join(root, ds, split) + '/', obs_len=obs_len, pred_len=pred_len, skip=1)
    scenes = []
    for i, (s, e) in enumerate(d.seq_start_end):
        obs, obs_rel = d.obs_traj[s:e], d.obs_traj_rel[s:e]
        _, A_ttc, E = seq_to_graph_hybrid(obs, obs_rel)
        scenes.append(dict(obs=obs, pred=d.pred_traj[s:e], obs_rel=obs_rel, pred_rel=d.pred_traj_rel[s:e],
                           V_obs=d.v_obs[i], V_pred=d.v_pred[i],
                           A_dist=d.A_obs[i], A_off=graph_official(obs_rel), A_ttc=A_ttc, E_obs=E))
    os.makedirs(CACHE, exist_ok=True)
    torch.save(scenes, path)
    return _add_graphs(scenes, need)


def _lean_split(data_dir, path, obs_len, pred_len):
    """Same windows and V as TrajectoryDataset, without building any adjacency (O(N^2) per window)."""
    import utils
    full = utils.seq_to_graph
    utils.seq_to_graph = lambda seq_, seq_rel, norm_lap_matr=True: (seq_rel.permute(2, 0, 1).float(), torch.zeros(0))
    try:
        d = TrajectoryDataset(data_dir, obs_len=obs_len, pred_len=pred_len, skip=1)
    finally:
        utils.seq_to_graph = full
    scenes = [dict(obs=d.obs_traj[s:e], pred=d.pred_traj[s:e], obs_rel=d.obs_traj_rel[s:e],
                   pred_rel=d.pred_traj_rel[s:e], V_obs=d.v_obs[i], V_pred=d.v_pred[i])
              for i, (s, e) in enumerate(d.seq_start_end)]
    os.makedirs(CACHE, exist_ok=True)
    torch.save(scenes, path)
    return scenes


def scene_graph(sc, graph):
    """The scene's cached graph, or the same graph computed on the fly (lean caches)."""
    if graph in sc:
        return sc[graph]
    from fast import batch_graphs
    n = sc['V_obs'].shape[1]
    return batch_graphs(sc['obs'].permute(2, 0, 1)[None], sc['V_obs'][None], torch.ones(1, n, dtype=torch.bool, device=sc['V_obs'].device), graph)[0]


def scene_energy(sc):
    if 'E_obs' in sc:
        return sc['E_obs']
    from fast import ttc_energy
    return ttc_energy(sc['obs'].permute(2, 0, 1).double(), sc['V_obs'].double()).float()


def make_model(obs_len=8, pred_len=12, n_stgcnn=1, n_txpcnn=5):
    return social_stgcnn(n_stgcnn=n_stgcnn, n_txpcnn=n_txpcnn, output_feat=5, seq_len=obs_len,
                         kernel_size=3, pred_seq_len=pred_len)


def forward(model, sc, graph):
    V = sc['V_obs'].unsqueeze(0).permute(0, 3, 1, 2)
    A = scene_graph(sc, graph)
    A_in = A.squeeze(0) if A.dim() == 4 else A
    if getattr(model, 'needs_x', False):
        out, _ = model(V, A_in, X=sc['obs'].permute(2, 0, 1).unsqueeze(0))
    else:
        out, _ = model(V, A_in)
    return out.permute(0, 2, 3, 1).squeeze(0)  # T,N,5


@torch.no_grad()
def evaluate(model, scenes, graph, k=20, seed=0, r_sum=0.4):
    """Best-of-k ADE/FDE (same as test.py) + collision metrics on the predicted mean."""
    torch.manual_seed(seed)
    model.eval()
    ades, fdes, col1, col1c, col2r, col2c, mind, c02, cs = [], [], [], [], [], [], [], [], []
    ae_sum, ae_M = 0.0, 0
    for sc in scenes:
        Vp = forward(model, sc, graph)
        N = Vp.shape[1]
        sx, sy, corr = torch.exp(Vp[..., 2]), torch.exp(Vp[..., 3]), torch.tanh(Vp[..., 4])
        cov = torch.stack([sx * sx, corr * sx * sy, corr * sx * sy, sy * sy], -1).view(*sx.shape, 2, 2)
        mean = Vp[..., :2]
        samp = torchdist.MultivariateNormal(mean, cov).sample((k,))  # k,T,N,2
        last = sc['obs'][:, :, -1]  # N,2
        gt = torch.cumsum(sc['V_pred'], 0) + last
        pa = torch.cumsum(samp, 1) + last
        err = (pa - gt).norm(dim=-1)  # k,T,N
        ades += err.mean(1).min(0).values.tolist()
        fdes += err[:, -1].min(0).values.tolist()
        if N >= 2:
            m = (torch.cumsum(mean, 0) + last).numpy()
            g = gt.numpy()
            obs_abs = sc['obs'].permute(2, 0, 1).numpy()
            gm = detect_group_pairs(obs_abs, sc['V_obs'].numpy())
            col1.append(col1_scene(m, r_sum))
            col1c += col1_check_scene(m, r_sum, gm).tolist()
            c02 += col1_check_scene(m, 0.2, gm).tolist()
            # joint samples: each of the k draws is one scene-level future
            dk = (pa[:, :, :, None] - pa[:, :, None]).norm(dim=-1).min(1).values  # k,N,N closest approach per sample
            dk[:, torch.from_numpy(np.eye(N, dtype=bool) | gm)] = float('inf')
            cs += (dk < 0.2).any(-1).float().mean(0).tolist()
            col2r.append(col2_rate_scene(m, g, r_sum))
            col2c += col2_check_scene(m, g, r_sum, gm).tolist()
            mind += min_dist_pairs_pred_pred(m, gm)
            ae_sum += avg_energy_scene(m, r_sum)
            ae_M += N
    T = scenes[0]['V_pred'].shape[0]
    return dict(ADE=float(np.mean(ades)), FDE=float(np.mean(fdes)), COL1=float(np.mean(col1)),
                COL1c=float(np.mean(col1c)), COL1c_r02=float(np.mean(c02)), COL1c_samp_r02=float(np.mean(cs)), COL2r=float(np.mean(col2r)), COL2c=float(np.mean(col2c)),
                AE=ae_sum / (ae_M * T), mind_p10=float(np.percentile(mind, 10)),
                mind_lt04=float(np.mean(np.asarray(mind) < r_sum)))


def gt_collision_stats(scenes, r_sum=0.4):
    """Collision rate of the ground truth itself under the same metric (the floor any predictor is judged against)."""
    col1c, mind = [], []
    for sc in scenes:
        if sc['obs'].shape[0] < 2:
            continue
        g = (torch.cumsum(sc['V_pred'], 0) + sc['obs'][:, :, -1]).numpy()
        gm = detect_group_pairs(sc['obs'].permute(2, 0, 1).numpy(), sc['V_obs'].numpy())
        col1c += col1_check_scene(g, r_sum, gm).tolist()
        mind += min_dist_pairs_pred_pred(g, gm)
    return dict(COL1c=float(np.mean(col1c)), mind_lt04=float(np.mean(np.asarray(mind) < r_sum)))
