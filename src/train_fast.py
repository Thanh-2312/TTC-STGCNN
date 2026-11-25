"""Packed-batch version of train_exp.py: 128 scenes per forward pass (= one optimizer step, as in train.py).
python src/train_fast.py --ds zara1 --graph A_off --loss nll --w 0 --seed 0 --txp 3 3 --out runs/x"""
import os, sys, copy, json, time, argparse, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_split, evaluate, scene_graph  # noqa: E402
from fast import FastSTGCNN, make_fast_model, batched_forward  # noqa: E402
from metrics import collision_loss, collision_loss_pred_gt  # noqa: E402


def nll_nodes(Vp, Vt):
    """bivariate_loss without the final mean: -log N per (t, node)."""
    nx, ny = Vt[..., 0] - Vp[..., 0], Vt[..., 1] - Vp[..., 1]
    sx, sy = torch.exp(Vp[..., 2].clamp(-20, 20)), torch.exp(Vp[..., 3].clamp(-20, 20))
    corr = torch.tanh(Vp[..., 4]).clamp(-0.999, 0.999)
    z = (nx / sx) ** 2 + (ny / sy) ** 2 - 2 * corr * nx * ny / (sx * sy)
    neg = 1 - corr ** 2
    p = torch.exp(-z / (2 * neg)) / (2 * np.pi * sx * sy * torch.sqrt(neg))
    return -torch.log(p.clamp(min=1e-20))


def scene_mean(x, seg, S):
    """x: per-node values (N,), seg: scene id -> mean over scenes of per-scene means."""
    s = torch.zeros(S, device=x.device).index_add(0, seg, x)
    c = torch.zeros(S, device=x.device).index_add(0, seg, torch.ones_like(x))
    return (s / c).mean()


def hinge_packed(Vp, scenes, seg, r=0.4):
    """Same as train_exp.hinge_loss averaged over scenes; pairs only within a scene (padded S,T,Nmax,Nmax)."""
    dev, S, T = Vp.device, len(scenes), Vp.shape[0]
    last = torch.cat([s['obs'][:, :, -1] for s in scenes])
    p = torch.cumsum(Vp[..., :2], 0) + last  # T,Nreal,2 (scene-major)
    n = torch.bincount(seg, minlength=S)
    start = torch.cumsum(n, 0) - n
    slot = torch.arange(len(seg), device=dev) - start[seg]  # index of each node within its scene
    Nmax = int(n.max())
    P = torch.zeros(S, Nmax, T, 2, device=dev)
    P[seg, slot] = p.permute(1, 0, 2)
    valid = torch.zeros(S, Nmax, dtype=torch.bool, device=dev)
    valid[seg, slot] = True
    P = P.permute(0, 2, 1, 3)  # S,T,Nmax,2
    d = (P.unsqueeze(3) - P.unsqueeze(2) + 1e-9).norm(dim=-1)  # S,T,Nmax,Nmax
    pair = valid[:, :, None] & valid[:, None, :] & ~torch.eye(Nmax, dtype=torch.bool, device=dev)
    pen = (torch.relu(r - d) * pair[:, None]).sum((1, 3))  # S,Nmax: per node, summed over t and partners
    per_node = pen / (T * (n[:, None] - 1).clamp(min=1))
    per_scene = (per_node * valid).sum(1) / n  # single-ped scenes give 0, as in train_exp
    return per_scene.mean()


def rsample(Vp, K=4):
    """K reparameterized draws of per-step displacements from the predicted bivariate Gaussians: (K,T,N,2)."""
    sx, sy = torch.exp(Vp[..., 2].clamp(-20, 20)), torch.exp(Vp[..., 3].clamp(-20, 20))
    rho = torch.tanh(Vp[..., 4]).clamp(-0.999, 0.999)
    z = torch.randn(K, *Vp.shape[:-1], 2, device=Vp.device)
    dx = Vp[..., 0] + sx * z[..., 0]
    dy = Vp[..., 1] + sy * (rho * z[..., 0] + torch.sqrt(1 - rho ** 2) * z[..., 1])
    return torch.stack([dx, dy], -1)


def aux(kind, Vp, scenes, seg):
    if kind == 'nll':
        return torch.zeros((), device=Vp.device)
    if kind == 'hinge':
        return hinge_packed(Vp, scenes, seg)
    if kind == 'hinge_samp':  # same penalty on joint samples, so gradients also shape sigma/rho
        return torch.stack([hinge_packed(x, scenes, seg) for x in rsample(Vp)]).mean()
    fn = {'ttc_pg': lambda v, s: collision_loss_pred_gt(v, s['V_pred'], s['obs']),
          'ttc_pp': lambda v, s: collision_loss(v, s['obs'])}[kind]
    out, p = [], 0
    for s in scenes:  # shortcut: per-scene loop for TTC losses; vectorize if TTC runs become the bottleneck
        n = s['obs'].shape[0]
        out.append(fn(Vp[:, p:p + n], s) if n > 1 else torch.zeros((), device=Vp.device))
        p += n
    return torch.stack(out).mean()


def permute_scene(s, graph):
    """Random reordering of the pedestrians of one scene (all node-indexed tensors consistently)."""
    pm = torch.randperm(s['V_obs'].shape[1], device=s['V_obs'].device)
    out = {'V_obs': s['V_obs'][:, pm], 'V_pred': s['V_pred'][:, pm], 'obs': s['obs'][pm]}
    if graph in s:  # lean scenes compute the graph from the permuted inputs instead
        out[graph] = s[graph][:, pm][:, :, pm]
    return out


GRADLOG = (0, 1, 2, 5, 10, 25, 50, 100, 150, 200)


class Symmetrized(torch.nn.Module):
    """Test-time group averaging: mean of the raw outputs over M fixed random re-indexings (invariant in expectation)."""
    def __init__(self, model, M=8, seed=0):
        super().__init__()
        self.model, self.M, self.seed = model, M, seed

    def forward(self, v, a, mask=None):
        g, N, outs = torch.Generator().manual_seed(self.seed), v.shape[-1], []
        for _ in range(self.M):
            pm = torch.randperm(N, generator=g)
            o, _ = self.model(v[..., pm], a[:, pm][:, :, pm])
            outs.append(o[..., torch.argsort(pm)])
        return torch.stack(outs).mean(0), a


@torch.no_grad()
def order_shift(model, scenes, graph, order='id', n_perm=3):
    """Mean shift of the predicted final position when the input pedestrians are re-indexed (scenes with N >= 3).
    For the position-based orders the pipeline re-canonicalises the permuted scene, as it would at test time."""
    model.eval()
    rng, shift = np.random.RandomState(0), []

    def pred(s):  # predicted mean displacements summed to the final step, in the scene's own order
        pm = order_perm(s, order, 0) if order in ('x', 'nn_chain') else torch.arange(s['V_obs'].shape[1])
        A_s = scene_graph(s, graph)
        v_in = s['V_obs'][:, pm].unsqueeze(0).permute(0, 3, 1, 2)
        a_in = A_s[:, pm][:, :, pm]
        if getattr(model, 'needs_x', False):
            out, _ = model(v_in, a_in, X=s['obs'][pm].permute(2, 0, 1).unsqueeze(0))
        else:
            out, _ = model(v_in, a_in)
        return out.permute(0, 2, 3, 1)[0, ..., :2].sum(0)[torch.argsort(pm)]

    for s in [s for s in scenes if s['obs'].shape[0] >= 3]:
        base = pred(s)
        for _ in range(n_perm):
            pm = torch.from_numpy(rng.permutation(base.shape[0]))
            sp = {k: (v[:, pm] if k in ('V_obs', 'V_pred') else v[pm] if k in ('obs', 'pred', 'obs_rel', 'pred_rel')
                      else v[:, pm][:, :, pm] if (k.startswith('A_') or k == 'E_obs') else v) for k, v in s.items()}
            shift += (pred(sp)[torch.argsort(pm)] - base).norm(dim=-1).tolist()
    return float(np.mean(shift))


def grad_share(model, scenes, graph, batches):
    """||grad L_pg|| / ||grad NLL|| and their cosine on fixed training batches in train mode, as in a real step
    (red-team RT6). Works on a copy so BatchNorm running statistics are untouched."""
    m = copy.deepcopy(model).train()
    params = list(m.parameters())
    out = []
    for b in batches:
        chunk = [scenes[i] for i in b]
        Vp, seg = batched_forward(m, chunk, graph)
        Vt = torch.cat([s['V_pred'] for s in chunk], 1)
        nll = scene_mean(nll_nodes(Vp, Vt).mean(0), seg, len(chunk))
        pg = aux('ttc_pg', Vp, chunk, seg)
        g1 = torch.autograd.grad(nll, params, retain_graph=True, allow_unused=True)  # the last PReLU is never used
        g2 = torch.autograd.grad(pg, params, allow_unused=True)
        flat = lambda gs: torch.cat([(torch.zeros_like(p) if x is None else x).flatten() for x, p in zip(gs, params)])
        f1, f2 = flat(g1), flat(g2)
        out.append(dict(ratio=(f2.norm() / f1.norm().clamp(min=1e-12)).item(),
                        cos=torch.nn.functional.cosine_similarity(f1, f2, dim=0).item(), nll=nll.item(), pg=pg.item()))
    return out


def order_perm(s, kind, i):
    """Index order of a scene's pedestrians: fixed_random (fixed per scene), reverse, x (sorted by last observed x)
    or nn_chain (nearest-neighbour chain from the leftmost pedestrian). x and nn_chain depend only on positions."""
    N, p = s['V_obs'].shape[1], s['obs'][:, :, -1].cpu()
    if kind == 'fixed_random':
        pm = torch.from_numpy(np.random.RandomState(100003 + i).permutation(N))
    elif kind == 'reverse':
        pm = torch.arange(N - 1, -1, -1)
    elif kind == 'x':
        pm = torch.argsort(p[:, 0])
    elif kind == 'nn_chain':
        D, cur = torch.cdist(p, p), int(torch.argmin(p[:, 0]))
        seq, left = [cur], set(range(N)) - {cur}
        while left:
            cand = torch.tensor(sorted(left))
            cur = int(cand[torch.argmin(D[cur, cand])])
            seq.append(cur); left.discard(cur)
        pm = torch.tensor(seq)
    else:
        raise ValueError(kind)
    return pm


def reorder(s, kind, i):
    """Re-index one scene's pedestrians once (red-team RT8) instead of the loader's sorted-track-id order."""
    pm = order_perm(s, kind, i).to(s['V_obs'].device)
    out = {}
    for k, v in s.items():
        if k in ('V_obs', 'V_pred'):
            out[k] = v[:, pm]
        elif k in ('obs', 'pred', 'obs_rel', 'pred_rel'):
            out[k] = v[pm]
        elif k.startswith('A_') or k == 'E_obs':
            out[k] = v[:, pm][:, :, pm]
        else:
            out[k] = v
    return out


def run_epoch(model, scenes, graph, kind, w, opt=None, bs=128, perm_aug=False):
    train = opt is not None
    model.train(train)
    order = np.random.permutation(len(scenes)) if train else np.arange(len(scenes))
    tot = nll_t = aux_t = 0.0
    with torch.set_grad_enabled(train):
        for b in range(0, len(order), bs):
            chunk = [scenes[i] for i in order[b:b + bs]]
            if train and perm_aug:
                chunk = [permute_scene(c, graph) for c in chunk]
            Vp, seg = batched_forward(model, chunk, graph)
            Vt = torch.cat([s['V_pred'] for s in chunk], 1)
            nll = scene_mean(nll_nodes(Vp, Vt).mean(0), seg, len(chunk))
            a = aux(kind, Vp, chunk, seg)
            loss = nll + w * a
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            k = len(chunk)
            tot += loss.item() * k; nll_t += nll.item() * k; aux_t += a.item() * k
    n = len(scenes)
    return tot / n, nll_t / n, aux_t / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--arch', default='stgcnn',
                    choices=['stgcnn', 'tf_global', 'tf_world', 'tf_pe', 'tf_rad1', 'tf_rad2', 'tf_cone1'])
    ap.add_argument('--graph', default='A_off')
    ap.add_argument('--loss', default='nll')
    ap.add_argument('--w', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--lr_step', type=int, default=150)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--txp', type=int, nargs=2, default=[3, 3])
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--perm_aug', action='store_true')
    ap.add_argument('--gradlog', action='store_true', help='log the TTC pred-GT gradient share during training')
    ap.add_argument('--order', default='id', help='pedestrian order of every scene (train/val/test): id = loader order')
    ap.add_argument('--pool', action='store_true', help='symmetric (mean-pool) interaction in the TXP head')
    ap.add_argument('--tta', type=int, default=0, help='also evaluate the test-time average over this many re-indexings')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    keep = ('V_obs', 'V_pred', 'obs', a.graph)  # only what training touches goes to the device
    tr, va = ([{k: s[k].to(dev) for k in keep if k in s} for s in load_split(a.ds, sp, need=(a.graph,))]
              for sp in ('train', 'val'))
    te = load_split(a.ds, 'test', need=(a.graph,))
    for split in (tr, va, te):  # lean caches: compute the graph once per scene instead of once per batch
        if split and a.graph not in split[0]:
            for s in split:
                s[a.graph] = scene_graph(s, a.graph).to(s['V_obs'].device)
    if a.order != 'id':
        tr, va, te = ([reorder(s, a.order, i) for i, s in enumerate(x)] for x in (tr, va, te))
    sizes = {sp: (len(x), max(s['V_obs'].shape[1] for s in x), float(np.mean([s['V_obs'].shape[1] for s in x])))
             for sp, x in (('train', tr), ('val', va), ('test', te))}
    print('scenes (count, max peds, mean peds):', sizes, flush=True)
    model = make_fast_model(arch=a.arch, txp_kernel=a.txp, pool=a.pool).to(dev)
    opt = (torch.optim.Adam(model.parameters(), lr=1e-3 if a.lr == 0.01 else a.lr) if a.arch != 'stgcnn'
           else torch.optim.SGD(model.parameters(), lr=a.lr))
    sch = torch.optim.lr_scheduler.StepLR(opt, a.lr_step, 0.2)
    best, hist, ck, t_start = float('inf'), [], os.path.join(a.out, 'val_best.pth'), time.time()
    perm = np.random.RandomState(1234).permutation(len(tr))
    gl_batches, glog = [perm[i * a.bs:(i + 1) * a.bs] for i in range(3)], []
    for ep in range(a.epochs):
        if a.gradlog and ep in GRADLOG:
            glog.append(dict(ep=ep, batches=grad_share(model, tr, a.graph, gl_batches)))
        t0 = time.time()
        trl = run_epoch(model, tr, a.graph, a.loss, a.w, opt, a.bs, a.perm_aug)
        val = run_epoch(model, va, a.graph, a.loss, a.w, bs=a.bs)
        sch.step()
        hist.append(dict(ep=ep, train=trl, val=val, sec=time.time() - t0))
        if val[0] < best:
            best = val[0]
            torch.save(model.state_dict(), ck)
        if ep % 10 == 0:
            print(hist[-1], flush=True)
    if a.gradlog:
        glog.append(dict(ep=a.epochs, batches=grad_share(model, tr, a.graph, gl_batches)))
    model = model.cpu()
    model.load_state_dict(torch.load(ck, map_location='cpu'))
    res = dict(vars(a), best_val=best, train_min=(time.time() - t_start) / 60, sizes=sizes,
               n_params=sum(p.numel() for p in model.parameters()), **evaluate(model, te, a.graph))
    if a.gradlog:
        res['grad_share'] = glog
    if a.order in ('id', 'x', 'nn_chain'):  # position-based or loader order: the full pipeline can be re-indexed
        res['order_shift'] = order_shift(model, te, a.graph, a.order)
    if a.tta:
        sym = Symmetrized(model, a.tta)
        res['tta_eval'] = dict(M=a.tta, order_shift=order_shift(sym, te, a.graph), **evaluate(sym, te, a.graph))
    json.dump(res, open(os.path.join(a.out, 'result.json'), 'w'), indent=1)
    json.dump(hist, open(os.path.join(a.out, 'history.json'), 'w'))
    print(res)


if __name__ == '__main__':
    main()
