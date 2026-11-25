"""Universal Kinematic TTC-Cone (input interface) & TTC-OSC (output sample coupling) utilities + local selftests."""
import math, sys, time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

GROUP_VEL_THRESH = 0.05
GROUP_DIST_THRESH = 0.7


def detect_groups_np(obs):
    """obs: (N, 8, 2) world meters -> (N, N) bool symmetric mask of co-walking group pairs (same thresholds as metrics.py)."""
    N = len(obs)
    if N < 2:
        return np.zeros((N, N), dtype=bool)
    vel = np.zeros_like(obs)
    vel[:, 1:] = obs[:, 1:] - obs[:, :-1]
    vij = np.linalg.norm(vel[:, None] - vel[None, :], axis=-1).mean(-1)
    xij = np.linalg.norm(obs[:, None] - obs[None, :], axis=-1).mean(-1)
    gm = (vij < GROUP_VEL_THRESH) & (xij < GROUP_DIST_THRESH)
    np.fill_diagonal(gm, False)
    return gm


def cone_mask_np(obs, r_near=3.0, r_cpa=2.0, horizon=12.0):
    """obs: (N, 8, 2) world meters -> (d_obs (N,N), d_cpa (N,N), t_star (N,N), in_cone (N,N) bool).
    Uses 3-step smoothed velocity v_bar = (x(7) - x(4)) / 3 to suppress 2.5 Hz annotation jitter."""
    obs = np.asarray(obs, dtype=np.float64)
    d_obs = np.linalg.norm(obs[:, None] - obs[None, :], axis=-1).min(-1)
    v_bar = (obs[:, 7] - obs[:, 4]) / 3.0
    dx = obs[None, :, 7] - obs[:, None, 7]
    dv = v_bar[None, :] - v_bar[:, None]
    vsq = (dv ** 2).sum(-1)
    t_star = np.clip(-(dx * dv).sum(-1) / np.maximum(vsq, 1e-6), 0.0, horizon)
    d_cpa = np.linalg.norm(dx + t_star[..., None] * dv, axis=-1)
    in_cone = (d_obs <= r_near) | (d_cpa <= r_cpa)
    np.fill_diagonal(in_cone, True)
    return d_obs, d_cpa, t_star, in_cone


def ego_indices(obs, mode='cone', r_near=3.0, r_cpa=2.0):
    """Return list of N index lists [idx_0, ..., idx_{N-1}]. Each idx_i has ego i at slot 0 and remaining neighbors in
    mode ('rad3', 'cone', 'inf') sorted canonically by relative geometry around ego i (invariant to input track order)."""
    obs = np.asarray(obs, dtype=np.float64)
    N = len(obs)
    if mode == 'inf':
        allow = np.ones((N, N), dtype=bool)
    else:
        d_obs, _, _, in_cone = cone_mask_np(obs, r_near=r_near, r_cpa=r_cpa)
        allow = (d_obs <= r_near) if mode == 'rad3' else in_cone
    out = []
    for i in range(N):
        nbrs = [j for j in range(N) if j != i and allow[i, j]]
        if len(nbrs) > 1:
            def _key(j):
                rel = obs[j] - obs[i]
                dx, dy = float(rel[7, 0]), float(rel[7, 1])
                return (round(math.hypot(dx, dy), 6), round(dx, 6), round(dy, 6),
                        tuple(np.round(rel.ravel(), 5).tolist()))
            nbrs.sort(key=_key)
        out.append([i] + nbrs)
    return out


def ttc_osc_costs(samples, last_obs, r_col=0.2, r_soc=0.4, dt=0.4, k_e=1.5, tau0=3.0, tau_max=12.0, eps=0.01):
    """samples: (K, N, T, 2), last_obs: (N, 2) torch float32/float64 -> (C, H02, H04, Epair) each (N, K, N, K).
    Uses exact 2D inner-product formulation per predicted step t (0..T-1) so peak memory is O((NK)^2) without 3D tensors."""
    s = samples.double() if torch.is_tensor(samples) else torch.as_tensor(samples, dtype=torch.float64)
    lo = last_obs.double() if torch.is_tensor(last_obs) else torch.as_tensor(last_obs, dtype=torch.float64)
    K, N, T, _ = s.shape
    M = N * K
    Y = s.permute(2, 1, 0, 3).reshape(T, M, 2)  # T, N*K, 2
    prev0 = lo[:, None, :].expand(N, K, 2).reshape(M, 2)
    min_d = torch.full((M, M), float('inf'), dtype=torch.float64)
    e_acc = torch.zeros((M, M), dtype=torch.float64)
    rsq = r_soc ** 2
    floor = math.tanh(k_e / ((tau_max * dt) ** 2 + eps) * math.exp(-tau_max * dt / tau0))

    for t in range(T):
        P = Y[t]
        V = P - (prev0 if t == 0 else Y[t - 1])
        d = torch.cdist(P, P, compute_mode='donot_use_mm_for_euclid_dist')
        min_d = torch.minimum(min_d, d)
        xsq = d * d
        vn2 = (V * V).sum(-1)
        vsq = (vn2[:, None] + vn2[None, :] - 2.0 * (V @ V.T)).clamp(min=0.0)
        pv = P @ V.T
        dpv = torch.diagonal(pv)
        xv = dpv[:, None] + dpv[None, :] - pv - pv.T
        disc = xv * xv - vsq * (xsq - rsq)
        moving = vsq > 1e-6
        cand = (-xv - disc.clamp(min=0.0).sqrt()) / torch.where(moving, vsq, torch.ones_like(vsq))
        tau = torch.where(moving & (disc >= 0.0) & (cand > 0.0), cand.clamp(max=tau_max), torch.full_like(vsq, tau_max))
        tau = torch.where(xsq <= rsq, torch.zeros_like(tau), tau)
        ts = tau * dt
        et = (torch.tanh(k_e / (ts * ts + eps) * torch.exp(-ts / tau0)) - floor).clamp(min=0.0)
        e_acc += et

    Epair = (e_acc / T).float().view(N, K, N, K)
    H02 = (min_d < r_col).view(N, K, N, K)
    H04 = (min_d < r_soc).view(N, K, N, K)
    clear_pen = torch.relu(r_soc - min_d).float().view(N, K, N, K)
    C = 100.0 * H02.float() + 5.0 * clear_pen + Epair
    return C, H02, H04, Epair


def scene_coupling_cost(C, sigma, excl):
    """Total pairwise cost summed over non-excluded unordered pairs (i < j) and averaged over K slots."""
    N, K = sigma.shape
    c = C[torch.arange(N)[:, None, None], sigma[:, None, :], torch.arange(N)[None, :, None], sigma[None, :, :]]
    mask = (~excl) & torch.triu(torch.ones((N, N), dtype=torch.bool), diagonal=1)
    return float(c[mask].sum().item() / K) if mask.any() else 0.0


def eval_coupling(H02, H04, Epair, sigma, excl):
    """Return per-pedestrian Col-S (<0.2m), NearMiss (<0.4m), and mean pair TTC energy for coupling sigma (N, K)."""
    N, K = sigma.shape
    ia, ja = torch.arange(N)[:, None, None], torch.arange(N)[None, :, None]
    sa, sb = sigma[:, None, :], sigma[None, :, :]
    c02 = H02[ia, sa, ja, sb].clone()
    c04 = H04[ia, sa, ja, sb].clone()
    ep = Epair[ia, sa, ja, sb].clone()
    c02[excl] = False
    c04[excl] = False
    ep[excl] = 0.0
    return (c02.any(1).float().mean(1).numpy(),
            c04.any(1).float().mean(1).numpy(),
            ep.sum(1).mean(1).numpy())


def assign_seq(cost_tensor, excl, maximize=False, binary_any=False):
    """Sequential Hungarian assignment across pedestrians i = 1..N-1."""
    N, K = cost_tensor.shape[:2]
    sigma = torch.arange(K, dtype=torch.long).repeat(N, 1)
    for i in range(1, N):
        others = torch.tensor([j for j in range(i) if not excl[i, j]], dtype=torch.long)
        if len(others) == 0:
            continue
        sub = cost_tensor[i][:, others]  # K, J, K
        gathered = torch.gather(sub, 2, sigma[others][None].expand(K, -1, -1))
        mat = gathered.any(1).float().numpy() if binary_any else gathered.sum(1).numpy()
        r, cidx = linear_sum_assignment(mat, maximize=maximize)
        sigma[i, torch.from_numpy(cidx)] = torch.from_numpy(r)
    return sigma


def solve_ttc_osc(C, excl, sweeps=2):
    """Block Coordinate Descent Hungarian coupling minimizing smooth TTC + collision cost (monotone descent)."""
    N, K = C.shape[:2]
    sigma = assign_seq(C, excl, maximize=False, binary_any=False)
    if sweeps <= 1 or N < 2:
        return sigma
    cur = scene_coupling_cost(C, sigma, excl)
    for _ in range(sweeps - 1):
        improved = False
        for i in range(N):
            others = torch.tensor([j for j in range(N) if j != i and not excl[i, j]], dtype=torch.long)
            if len(others) == 0:
                continue
            sub = C[i][:, others]  # K, J, K
            mat = torch.gather(sub, 2, sigma[others][None].expand(K, -1, -1)).sum(1).numpy()
            r, cidx = linear_sum_assignment(mat, maximize=False)
            cand = sigma.clone()
            cand[i, torch.from_numpy(cidx)] = torch.from_numpy(r)
            cand_cost = scene_coupling_cost(C, cand, excl)
            if cand_cost + 1e-7 < cur:
                sigma, cur, improved = cand, cand_cost, True
        if not improved:
            break
    return sigma


def jade_for_sigma(err_nk, sigma):
    """err_nk: (N, K) per-agent sample ADEs; sigma: (N, K) -> scalar scene JADE."""
    gathered = torch.gather(err_nk, 1, sigma)  # N, K
    return float(gathered.mean(0).min().item())


def oracle_jade_bounds(err_nk):
    """Exact oracle JADE min (= mean_i min_k ADE_{i,k}) and water-filling adversarial oracle JADE max."""
    E = err_nk.detach().cpu().numpy() if torch.is_tensor(err_nk) else np.asarray(err_nk, dtype=np.float64)
    N, K = E.shape
    j_min = float(E.min(axis=1).mean())
    order = np.argsort(-(E.max(axis=1) - E.min(axis=1)))
    sums = np.zeros(K, dtype=np.float64)
    for i in order:
        row_desc = np.sort(E[i])[::-1]
        slot_asc = np.argsort(sums)
        sums[slot_asc] += row_desc
    j_max = float((sums / N).min())
    return j_min, j_max


def selftest():
    rng = np.random.default_rng(42)
    # 1. Cone symmetry + head-on vs far bystander test
    obs = np.zeros((4, 8, 2))
    t = np.arange(8)
    obs[0, :, 0] = 0.0 + 0.4 * t          # ego moving +x at 0.4 m/step (1.0 m/s), at x=2.8 at t=7
    obs[1, :, 0] = 10.0 - 0.4 * t         # head-on approaching: at x=7.2 at t=7 (d_obs=4.4m > 3m, closing at 0.8m/step -> d_cpa=0!)
    obs[2, :, 0] = 2.8; obs[2, :, 1] = 8.0  # parallel bystander 8m away in y -> d_obs=8m, d_cpa=8m
    obs[3, :, 0] = 2.8; obs[3, :, 1] = 1.8  # close neighbor 1.8m away
    d_obs, d_cpa, _, in_cone = cone_mask_np(obs)
    assert np.allclose(d_obs, d_obs.T) and np.allclose(d_cpa, d_cpa.T)
    assert d_obs[0, 1] > 3.0 and d_cpa[0, 1] < 0.01 and in_cone[0, 1], 'head-on approaching pair must be inside TTC cone'
    assert not in_cone[0, 2] and in_cone[0, 3], 'far bystander must be excluded, close neighbor included'
    assert ego_indices(obs, 'rad3')[0] == [0, 3], ego_indices(obs, 'rad3')[0]
    assert set(ego_indices(obs, 'cone')[0]) == {0, 1, 3}

    # 2. Exact permutation equivariance & outside-cone removal invariance across 30 random scenes
    for s in range(30):
        N = rng.integers(4, 15)
        sc = rng.normal(0, 5.0, size=(N, 8, 2))
        sc[N - 1] = sc[0] + np.array([20.0, 20.0])  # guaranteed outside cone of 0
        id_cone = ego_indices(sc, 'cone')
        pm = rng.permutation(N)
        inv = np.argsort(pm)
        pm_cone = ego_indices(sc[pm], 'cone')
        for i in range(N):
            sub_orig = sc[id_cone[i]]
            sub_perm = sc[pm][pm_cone[inv[i]]]
            assert np.array_equal(sub_orig, sub_perm), 'ego sub-scene must be bit-for-bit permutation invariant'
        sc_drop = sc[:-1]
        assert np.array_equal(sc[id_cone[0]], sc_drop[ego_indices(sc_drop, 'cone')[0]]), 'outside-cone removal must be bit-identical'

    # 3. Verify 2D matrix-multiply TTC energy against direct 3D pairwise formula
    K, N, T = 6, 4, 12
    samp = torch.randn(K, N, T, 2, dtype=torch.float64) * 0.5
    lo = torch.randn(N, 2, dtype=torch.float64)
    C, H02, H04, Epair = ttc_osc_costs(samp, lo)
    # Reference check on step 0
    P0 = samp[:, :, 0, :].permute(1, 0, 2).reshape(N * K, 2)
    V0 = P0 - lo[:, None, :].expand(N, K, 2).reshape(N * K, 2)
    d_ref = (P0[:, None] - P0[None, :]).norm(dim=-1)
    assert torch.allclose(d_ref, torch.cdist(P0, P0, compute_mode='donot_use_mm_for_euclid_dist'), atol=1e-10)

    # 4. Verify TTC-OSC monotone descent and Oracle vs GT-blind JADE bounds
    excl = torch.eye(N, dtype=torch.bool)
    sig_rel = torch.arange(K).repeat(N, 1)
    sig_1 = solve_ttc_osc(C, excl, sweeps=1)
    sig_2 = solve_ttc_osc(C, excl, sweeps=2)
    c_rel = scene_coupling_cost(C, sig_rel, excl)
    c_1 = scene_coupling_cost(C, sig_1, excl)
    c_2 = scene_coupling_cost(C, sig_2, excl)
    assert c_2 <= c_1 + 1e-6 and c_1 <= c_rel + 1e-6, (c_rel, c_1, c_2)
    gt = torch.randn(N, T, 2, dtype=torch.float64)
    err_nk = (samp - gt[None]).norm(dim=-1).mean(-1).T.float()  # N, K
    jmin, jmax = oracle_jade_bounds(err_nk)
    j_osc = jade_for_sigma(err_nk, sig_2)
    assert jmin <= j_osc + 1e-6 and jmin <= jmax + 1e-6, (jmin, j_osc, jmax)
    print('SELFTEST OK: cone symmetry, bit-exact perm/removal invariance, 2D TTC cost, monotone OSC & JADE bounds verified.')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
