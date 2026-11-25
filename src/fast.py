"""Packed-batch Social-STGCNN: many scenes in one forward pass.

Scenes are concatenated along the node axis with one zero gap node between them, A is block-diagonal, and the gap is
re-zeroed after every TXP-CNN layer, so each scene sees exactly the zero padding it sees alone. Parameter names match
model.social_stgcnn, so existing checkpoints load unchanged. txp_kernel=(3,1) removes the node-order-dependent mixing.
"""
import math
import torch
import torch.nn as nn
from model import social_stgcnn


class FastSTGCNN(social_stgcnn):
    def __init__(self, txp_kernel=(3, 3), obs_len=8, pred_len=12, n_stgcnn=1, n_txpcnn=5, pool=False):
        super().__init__(n_stgcnn=n_stgcnn, n_txpcnn=n_txpcnn, output_feat=5, seq_len=obs_len,
                         kernel_size=3, pred_seq_len=pred_len)
        # pool=True: symmetric interaction in the head, each TXP layer adds a 1x1 map of the mean over pedestrians
        self.pools = nn.ModuleList([nn.Conv2d(pred_len, pred_len, 1) for _ in range(n_txpcnn - 1)]) if pool else None
        if tuple(txp_kernel) != (3, 3):
            pad = tuple(k // 2 for k in txp_kernel)
            self.tpcnns = nn.ModuleList([nn.Conv2d(obs_len, pred_len, txp_kernel, padding=pad)] +
                                        [nn.Conv2d(pred_len, pred_len, txp_kernel, padding=pad) for _ in range(1, n_txpcnn)])
            self.tpcnn_ouput = nn.Conv2d(pred_len, pred_len, txp_kernel, padding=pad)

    def forward(self, v, a, mask=None, X=None):
        m = (lambda x: x) if mask is None else (lambda x: x * mask)
        for k in range(self.n_stgcnn):
            if a.dim() == 4:  # padded batch: one A per scene (S,T,N,N)
                g = self.st_gcns[k]
                res = g.residual(v)
                x = torch.einsum('nctv,ntvw->nctw', g.gcn.conv(v), a)
                v = g.prelu(g.tcn(x) + res)
            else:
                v, a = self.st_gcns[k](v, a)
        v = m(v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3]))
        if self.pools is None:
            pool = lambda k, x: x
        else:  # mean over the real pedestrians of each scene (padding excluded)
            mean = (lambda x: x.mean(-1, keepdim=True)) if mask is None else                 (lambda x: (x * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1))
            pool = lambda k, x: x + self.pools[k](mean(x))
        v = m(pool(0, self.prelus[0](self.tpcnns[0](v))))
        for k in range(1, self.n_txpcnn - 1):
            v = m(pool(k, self.prelus[k](self.tpcnns[k](v))) + v)
        v = self.tpcnn_ouput(v)
        return v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3]), a


def pack(scenes, graph):
    """-> V (1,2,T,Ntot), A (T,Ntot,Ntot), mask (Ntot,), idx (LongTensor of real-node positions), seg (scene id per real node)"""
    Vs, As, idx, seg, pos = [], [], [], [], 0
    for i, s in enumerate(scenes):
        n = s['V_obs'].shape[1]
        Vs += [s['V_obs'], torch.zeros(s['V_obs'].shape[0], 1, 2)]
        As.append(s[graph])
        idx.append(torch.arange(pos, pos + n)); seg.append(torch.full((n,), i))
        pos += n + 1
    V = torch.cat(Vs, 1)  # T,Ntot,2
    T, Nt = V.shape[0], V.shape[1]
    A = torch.zeros(T, Nt, Nt)
    p = 0
    for a in As:
        n = a.shape[1]
        A[:, p:p + n, p:p + n] = a
        p += n + 1
    idx = torch.cat(idx)
    mask = torch.zeros(Nt); mask[idx] = 1
    return V.permute(2, 0, 1).unsqueeze(0), A, mask, idx, torch.cat(seg)


def ttc_energy(X, V, r_sum=0.4, tau_max=12.0, k=1.5, tau0=3.0, eps=0.01, dt=0.4):
    """Interaction energy E (...,T,N,N), float64, same math as utils.seq_to_graph_hybrid; diagonal uses tau_max.
    X absolute positions, V per-step displacements, both (...,T,N,2)."""
    xij, vij = X.unsqueeze(-2) - X.unsqueeze(-3), V.unsqueeze(-2) - V.unsqueeze(-3)
    vsq, xsq, xv = (vij ** 2).sum(-1), (xij ** 2).sum(-1), (xij * vij).sum(-1)
    disc = xv ** 2 - vsq * (xsq - r_sum ** 2)
    moving = vsq > 1e-6
    cand = (-xv - disc.clamp(min=0).sqrt()) / torch.where(moving, vsq, torch.ones_like(vsq))
    tau = torch.where(moving & (disc >= 0) & (cand > 0), cand.clamp(max=tau_max), torch.full_like(vsq, tau_max))
    tau = torch.where(xsq <= r_sum ** 2, torch.zeros_like(tau), tau)
    eye = torch.eye(X.shape[-2], dtype=torch.bool, device=X.device)
    tau = torch.where(eye, torch.full_like(tau, tau_max), tau)
    ts = tau * dt
    return k / (ts ** 2 + eps) * torch.exp(-ts / tau0)


def batch_graphs(X, V, valid, kind):
    """Graphs for a padded batch, identical to the per-scene graphs of common.py / utils.py (float64 math, float32
    out) and zero on padded pedestrians. X, V: (S,T,N,2); valid: (S,N) bool. kind in A_off, A_ttc, A_eye, A_hyb."""
    X, V = X.double(), V.double()
    S, T, N, _ = V.shape
    pair = (valid[:, :, None] & valid[:, None, :])[:, None]  # S,1,N,N
    eye = torch.eye(N, dtype=torch.bool, device=V.device)
    dvalid = (eye & pair).double()
    if kind == 'A_eye':
        return dvalid.expand(S, T, N, N).float().contiguous()
    if kind in ('A_ttc', 'A_unif', 'A_ttc0'):
        t = torch.tanh(ttc_energy(X, V))
        floor = math.tanh(1.5 / ((12.0 * 0.4) ** 2 + 0.01) * math.exp(-12.0 * 0.4 / 3.0))  # tanh E(tau_max)
        if kind == 'A_unif':  # red-team control: only the constant floor every non-converging pair receives
            t = torch.full_like(t, floor)
        elif kind == 'A_ttc0':  # red-team control: the TTC graph without that floor
            t = (t - floor).clamp(min=0)
        a = (t + eye.double()) * pair
        dinv = torch.where(valid[:, None], a.sum(-1).clamp(min=1e-30).rsqrt(), torch.zeros((), dtype=a.dtype, device=a.device))
        return (dinv[..., :, None] * a * dinv[..., None, :]).float()
    # A_pos (J6): 1/distance between positions instead of between displacements; A_rad (J7): the same, zero beyond 3 m
    src = X if kind in ('A_pos', 'A_rad', 'A_radn', 'A_cone') else V
    d = (src.unsqueeze(-2) - src.unsqueeze(-3)).norm(dim=-1)
    a = torch.where(d > 0, 1.0 / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(d))
    if kind == 'A_hyb':
        a = a + torch.tanh(ttc_energy(X, V).float().double())  # graph_hybrid reads the float32 cached E_obs
    elif kind in ('A_rad', 'A_radn'):
        a = torch.where(d < 3.0, a, torch.zeros_like(a))
    elif kind == 'A_cone':
        d_min = d.min(dim=1, keepdim=True).values
        step_back = max(0, T - 4)
        denom_t = float(max(1, (T - 1) - step_back))
        v_bar = (X[:, -1] - X[:, step_back]) / denom_t
        dx = X[:, -1].unsqueeze(-2) - X[:, -1].unsqueeze(-3)
        dv = v_bar.unsqueeze(-2) - v_bar.unsqueeze(-3)
        vsq = (dv ** 2).sum(-1)
        t_star = (-(dx * dv).sum(-1) / vsq.clamp(min=1e-6)).clamp(0.0, 12.0)
        d_cpa = (dx + t_star.unsqueeze(-1) * dv).norm(dim=-1)
        in_cone = (d_min <= 3.0) | (d_cpa[:, None] <= 2.0)
        a_urg = torch.maximum(a, 1.0 / (1.0 + d_cpa[:, None]))
        a = torch.where(in_cone & (~eye[None, None]), a_urg, torch.zeros_like(a))
    elif kind not in ('A_off', 'A_pos'):
        raise ValueError(kind)
    a = torch.where(eye, torch.ones_like(a), a) * pair
    if kind in ('A_radn', 'A_cone'):  # receiver-only normalization: out_w = sum_v x_v A[v, w], divide by receiver degree only
        deg = torch.where(valid[:, None], a.sum(-2).clamp(min=1e-30), torch.ones((), dtype=a.dtype, device=a.device))
        return ((dvalid - a / deg[..., None, :]) * pair).float()
    dinv = torch.where(valid[:, None], a.sum(-1).clamp(min=1e-30).rsqrt(), torch.zeros((), dtype=a.dtype, device=a.device))
    return ((dvalid - dinv[..., :, None] * a * dinv[..., None, :]) * pair).float()


class _RelSocialAttnLayer(nn.Module):
    def __init__(self, d_model=32, nhead=4, use_rel=False):
        super().__init__()
        self.d_model, self.nhead, self.dk, self.use_rel = d_model, nhead, d_model // nhead, use_rel
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rel_logit = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, nhead)) if use_rel else None
        self.rel_val = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model)) if use_rel else None
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.SiLU(), nn.Linear(2 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, u, allow, rel_feat=None):
        S, N, D = u.shape
        qkv = self.qkv(u).view(S, N, 3, self.nhead, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # S, H, N, dk
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if self.use_rel and rel_feat is not None:
            scores = scores + self.rel_logit(rel_feat).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~allow[:, None], -1e9)
        alpha = torch.softmax(scores, dim=-1)
        out = torch.matmul(alpha, v).permute(0, 2, 1, 3).reshape(S, N, D)
        if self.use_rel and rel_feat is not None:
            w_mean = alpha.mean(dim=1, keepdim=False).unsqueeze(-1)  # S, N, N, 1
            out = out + (w_mean * self.rel_val(rel_feat)).sum(dim=2)
        u = self.norm1(u + self.proj(out))
        return self.norm2(u + self.ffn(u))


class SocialTransformerFast(nn.Module):
    """Controlled Social-Transformer counterpart to FastSTGCNN (identical 3x1 TXP-CNN horizon head & Gaussian output).
    Modes:
      tf_global : unmasked global Softmax on displacement tokens (Condition i violation -> flat global leak)
      tf_world  : unmasked Softmax + scene-centered world origin (Condition ii violation -> inverted Far > Near leak)
      tf_pe     : unmasked Softmax + sinusoidal agent-index PE (Axiom I violation -> order_shift > 0)
      tf_rad1   : 1-layer receiver-relative features + hard 3 m mask (exact 0 shift > 3 m)
      tf_rad2   : 2-layer receiver-relative features + hard 3 m mask (2-hop leak in 3-6 m, exact 0 shift > 6 m)
      tf_cone1  : 1-layer receiver-relative features + Kinematic TTC-Cone mask (exact 0 outside cone + approaching sensitivity)
    """
    needs_x = True

    def __init__(self, mode='tf_cone1', obs_len=8, pred_len=12, n_txpcnn=5, d_model=32, nhead=4):
        super().__init__()
        assert mode in ('tf_global', 'tf_world', 'tf_pe', 'tf_rad1', 'tf_rad2', 'tf_cone1'), mode
        self.mode, self.obs_len, self.pred_len, self.n_txpcnn, self.d_model = mode, obs_len, pred_len, n_txpcnn, d_model
        use_rel = mode in ('tf_rad1', 'tf_rad2', 'tf_cone1')
        n_layers = 2 if mode == 'tf_rad2' else 1
        self.temp_enc = nn.Sequential(
            nn.Conv1d(2, d_model, 3, padding=1), nn.SiLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.SiLU(),
        )
        self.world_proj = nn.Linear(2, d_model) if mode == 'tf_world' else None
        if mode == 'tf_pe':
            pos = torch.arange(256, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(256, d_model)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer('pe', pe)
        self.layers = nn.ModuleList([_RelSocialAttnLayer(d_model, nhead, use_rel=use_rel) for _ in range(n_layers)])
        self.to_feat = nn.Linear(d_model, 5)
        self.tpcnns = nn.ModuleList([nn.Conv2d(obs_len, pred_len, (3, 1), padding=(1, 0))] +
                                    [nn.Conv2d(pred_len, pred_len, (3, 1), padding=(1, 0)) for _ in range(1, n_txpcnn)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(n_txpcnn)])
        self.tpcnn_ouput = nn.Conv2d(pred_len, pred_len, (3, 1), padding=(1, 0))

    def forward(self, v, a=None, mask=None, X=None):
        # v: (S, 2, T, N), X: (S, T, N, 2)
        S, _, T, N = v.shape
        assert X is not None, 'SocialTransformerFast requires absolute positions X=(S,T,N,2)'
        if X.dim() == 3:
            X = X.unsqueeze(0)
        valid = torch.ones((S, N), dtype=torch.bool, device=v.device) if mask is None else (mask.view(S, N) > 0.5)
        m = (lambda z: z) if mask is None else (lambda z: z * mask)

        vt = v.permute(0, 3, 1, 2).reshape(S * N, 2, T)
        h = self.temp_enc(vt).reshape(S, N, self.d_model, T).permute(0, 3, 1, 2)  # S, T, N, D
        u = h.mean(dim=1)  # S, N, D

        if self.mode == 'tf_world':
            x0 = X[:, 0]  # S, N, 2
            vf = valid.unsqueeze(-1).float()
            x0_cen = (x0 - (x0 * vf).sum(dim=1, keepdim=True) / vf.sum(dim=1, keepdim=True).clamp(min=1.0)) * vf
            u = u + self.world_proj(x0_cen)
        elif self.mode == 'tf_pe':
            u = u + self.pe[:N].unsqueeze(0)

        eye = torch.eye(N, dtype=torch.bool, device=v.device)[None]
        allow = valid[:, :, None] & valid[:, None, :]
        rel_feat = None
        if self.mode in ('tf_rad1', 'tf_rad2', 'tf_cone1'):
            d_all = (X.unsqueeze(-2) - X.unsqueeze(-3)).norm(dim=-1)  # S, T, N, N
            d_obs = d_all.min(dim=1).values
            step_back = max(0, T - 4)
            v_bar = (X[:, -1] - X[:, step_back]) / float(max(1, (T - 1) - step_back))
            rel_x = X[:, -1].unsqueeze(-3) - X[:, -1].unsqueeze(-2)   # X_j - X_i: (S, N, N, 2)
            rel_v = v_bar.unsqueeze(-3) - v_bar.unsqueeze(-2)         # v_j - v_i: (S, N, N, 2)
            vsq = (rel_v ** 2).sum(-1)
            t_star = (-(rel_x * rel_v).sum(-1) / vsq.clamp(min=1e-6)).clamp(0.0, 12.0)
            d_cpa = (rel_x + t_star.unsqueeze(-1) * rel_v).norm(dim=-1)
            geom = ((d_obs <= 3.0) | (d_cpa <= 2.0)) if self.mode == 'tf_cone1' else (d_obs <= 3.0)
            allow = allow & geom
            rel_feat = torch.cat([rel_x, rel_v, 1.0 / (1.0 + d_obs.unsqueeze(-1)), 1.0 / (1.0 + d_cpa.unsqueeze(-1))], dim=-1)
        allow = allow | eye  # self-attention slot always open so softmax never sees an empty row

        for layer in self.layers:
            u = layer(u, allow, rel_feat) * valid.unsqueeze(-1).float()

        z = self.to_feat(h + u[:, None]).permute(0, 1, 3, 2)  # S, T, 5, N
        z = m(z)
        z = m(self.prelus[0](self.tpcnns[0](z)))
        for k in range(1, self.n_txpcnn - 1):
            z = m(self.prelus[k](self.tpcnns[k](z)) + z)
        z = self.tpcnn_ouput(z)
        return z.permute(0, 2, 1, 3), a


def make_fast_model(arch='stgcnn', txp_kernel=(3, 1), pool=False, obs_len=8, pred_len=12, n_stgcnn=1, n_txpcnn=5):
    if arch == 'stgcnn':
        return FastSTGCNN(txp_kernel=txp_kernel, obs_len=obs_len, pred_len=pred_len,
                          n_stgcnn=n_stgcnn, n_txpcnn=n_txpcnn, pool=pool)
    return SocialTransformerFast(mode=arch, obs_len=obs_len, pred_len=pred_len, n_txpcnn=n_txpcnn)


def pad_batch(scenes, graph):
    """-> V (S,2,T,Nmax), A (S,T,Nmax,Nmax), mask (S,1,1,Nmax), seg (scene id per real node, scene-major)
    Scenes without a cached `graph` (lean SDD cache) get it computed for the whole batch by batch_graphs."""
    S, Nmax, T = len(scenes), max(s['V_obs'].shape[1] for s in scenes), scenes[0]['V_obs'].shape[0]
    dev = scenes[0]['V_obs'].device
    V = torch.zeros(S, T, Nmax, 2, device=dev)
    mask = torch.zeros(S, Nmax, dtype=torch.bool, device=dev)
    cached = graph in scenes[0]
    if cached:
        A = torch.zeros(S, T, Nmax, Nmax, device=dev)
    else:
        X = torch.zeros(S, T, Nmax, 2, device=dev)
    for i, s in enumerate(scenes):
        n = s['V_obs'].shape[1]
        V[i, :, :n] = s['V_obs']; mask[i, :n] = True
        if cached:
            A[i, :, :n, :n] = s[graph]
        else:
            X[i, :, :n] = s['obs'].permute(2, 0, 1)
    if not cached:
        A = batch_graphs(X, V, mask, graph)
    seg = torch.arange(S, device=dev)[:, None].expand(S, Nmax)[mask]
    return V.permute(0, 3, 1, 2), A, mask, seg


def batched_forward(model, scenes, graph):
    """Padded-batch pass. Returns (T, Nreal, 5) in scene-major node order, and seg."""
    V, A, mask, seg = pad_batch(scenes, graph)
    X = None
    if getattr(model, 'needs_x', False):
        S, T, Nmax = len(scenes), V.shape[2], V.shape[3]
        X = torch.zeros(S, T, Nmax, 2, device=V.device)
        for i, s in enumerate(scenes):
            n = s['obs'].shape[0]
            X[i, :, :n] = s['obs'].permute(2, 0, 1)
    out, _ = model(V, A, mask[:, None, None, :].float(), X=X) if X is not None else model(V, A, mask[:, None, None, :].float())
    return out.permute(0, 2, 3, 1).permute(1, 0, 2, 3)[:, mask], seg


def packed_forward(model, scenes, graph):
    """Returns per-scene outputs (list of T,N,5) from one packed pass."""
    V, A, mask, idx, seg = pack(scenes, graph)
    out, _ = model(V, A, mask)
    out = out.permute(0, 2, 3, 1)[0][:, idx]  # T, Nreal, 5
    return out, seg


def ttc_graph_fast(obs, obs_rel, r_sum=0.4, tau_max=12.0, k=1.5, tau0=3.0, eps=0.01, dt=0.4):
    """Vectorized over all timesteps; same math as utils.seq_to_graph_hybrid (A only). obs, obs_rel: (N,2,T)."""
    x = obs.permute(2, 0, 1).double()   # T,N,2
    v = obs_rel.permute(2, 0, 1).double()
    xij, vij = x[:, :, None] - x[:, None], v[:, :, None] - v[:, None]
    vsq, xsq, xv = (vij ** 2).sum(-1), (xij ** 2).sum(-1), (xij * vij).sum(-1)
    disc = xv ** 2 - vsq * (xsq - r_sum ** 2)
    moving = vsq > 1e-6
    cand = (-xv - disc.clamp(min=0).sqrt()) / torch.where(moving, vsq, torch.ones_like(vsq))
    tau = torch.where(moving & (disc >= 0) & (cand > 0), cand.clamp(max=tau_max), torch.full_like(vsq, tau_max))
    tau = torch.where(xsq <= r_sum ** 2, torch.zeros_like(tau), tau)
    N = x.shape[1]
    eye = torch.eye(N, dtype=torch.bool)
    tau = torch.where(eye, torch.full_like(tau, tau_max), tau)
    ts = tau * dt
    A = torch.tanh(k / (ts ** 2 + eps) * torch.exp(-ts / tau0)) + torch.eye(N)
    dinv = A.sum(-1).rsqrt()
    return (dinv[..., :, None] * A * dinv[..., None, :]).float()
