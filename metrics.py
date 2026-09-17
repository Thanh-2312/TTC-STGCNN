import math
import numpy as np
import torch
import matplotlib.pyplot as plt

PED_RADIUS = 0.2
dt = 0.4
ENERGY_K = 1.5
ENERGY_TAU0 = 3.0
TAU_MAX = 12.0
EPS_ENERGY = 0.01
BETA_PRED_GT = 1.0
GROUP_VEL_THRESH = 0.05
GROUP_DIST_THRESH = 0.7


def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0])
                                  ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0])
                                  ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / N

    return sum_all / All


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(
                nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]

    return nodes_.squeeze()


def bivariate_loss(V_pred, V_trgt):
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(torch.clamp(V_pred[:, :, 2], min=-20, max=20))
    sy = torch.exp(torch.clamp(V_pred[:, :, 3], min=-20, max=20))

    corr = torch.tanh(V_pred[:, :, 4])
    corr = torch.clamp(corr, min=-0.999, max=0.999)

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - \
        2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    result = torch.exp(-z / (2 * negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    result = result / denom
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result


def pairwise_ttc_torch(pos, vel, r_sum, tau_max=TAU_MAX, eps=1e-6):
    x_ij = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    v_ij = vel.unsqueeze(-2) - vel.unsqueeze(-3)

    v_sq = (v_ij ** 2).sum(-1)
    x_sq = (x_ij ** 2).sum(-1)
    x_dot_v = (x_ij * v_ij).sum(-1)

    disc = x_dot_v ** 2 - v_sq * (x_sq - r_sum ** 2)

    moving = v_sq > eps
    real_roots = disc >= 0

    safe_vsq = torch.where(moving, v_sq, torch.ones_like(v_sq))
    safe_disc = torch.clamp(disc, min=1e-8)
    sqrt_disc = torch.sqrt(safe_disc)

    candidate = (-x_dot_v - sqrt_disc) / safe_vsq

    tau_max_t = torch.full_like(v_sq, tau_max)
    valid_future = moving & real_roots & (candidate > 0)
    tau = torch.where(valid_future, torch.clamp(
        candidate, max=tau_max), tau_max_t)

    already_colliding = x_sq <= (r_sum ** 2)
    tau = torch.where(already_colliding, torch.zeros_like(tau), tau)

    N = pos.shape[-2]
    eye = torch.eye(N, dtype=torch.bool, device=pos.device)
    tau = torch.where(eye, tau_max_t, tau)
    return tau


def interaction_energy_torch(tau, k=ENERGY_K, tau0=ENERGY_TAU0, eps=EPS_ENERGY):
    tau_sec = tau * dt
    E = (k / (tau_sec ** 2 + eps)) * torch.exp(-tau_sec / tau0)
    return E


def collision_loss(V_pred, obs_traj_batch, r_sum=2 * PED_RADIUS, k=ENERGY_K, tau0=ENERGY_TAU0, tau_max=TAU_MAX):
    mu = V_pred[..., 0:2]

    obs = obs_traj_batch
    while obs.dim() > 3:
        obs = obs.squeeze(0)
    last_obs_pos = obs[:, :, -1].detach()

    abs_pred = torch.cumsum(mu, dim=0) + last_obs_pos.unsqueeze(0)

    tau = pairwise_ttc_torch(abs_pred, mu, r_sum, tau_max)
    E = interaction_energy_torch(tau, k, tau0)

    N = mu.shape[1]
    eye = torch.eye(N, dtype=torch.bool, device=mu.device)
    mask = (~eye).float()
    denom = mask.sum().clamp(min=1.0)

    per_t = (torch.tanh(E) * mask).sum(dim=(-2, -1)) / denom
    L_collision = per_t.mean()
    return L_collision


def pairwise_ttc_torch_cross(pos_a, vel_a, pos_b, vel_b, r_sum, tau_max=TAU_MAX, eps=1e-6):
    x_ij = pos_a.unsqueeze(-2) - pos_b.unsqueeze(-3)
    v_ij = vel_a.unsqueeze(-2) - vel_b.unsqueeze(-3)

    v_sq = (v_ij ** 2).sum(-1)
    x_sq = (x_ij ** 2).sum(-1)
    x_dot_v = (x_ij * v_ij).sum(-1)

    disc = x_dot_v ** 2 - v_sq * (x_sq - r_sum ** 2)

    moving = v_sq > eps
    real_roots = disc >= 0

    safe_vsq = torch.where(moving, v_sq, torch.ones_like(v_sq))
    safe_disc = torch.clamp(disc, min=1e-8)
    sqrt_disc = torch.sqrt(safe_disc)

    candidate = (-x_dot_v - sqrt_disc) / safe_vsq

    tau_max_t = torch.full_like(v_sq, tau_max)
    valid_future = moving & real_roots & (candidate > 0)
    tau = torch.where(valid_future, torch.clamp(
        candidate, max=tau_max), tau_max_t)

    already_colliding = x_sq <= (r_sum ** 2)
    tau = torch.where(already_colliding, torch.zeros_like(tau), tau)

    N_a = pos_a.shape[-2]
    N_b = pos_b.shape[-2]
    if N_a == N_b:
        eye = torch.eye(N_a, dtype=torch.bool, device=pos_a.device)
        tau = torch.where(eye, tau_max_t, tau)
    return tau


def collision_loss_pred_gt(V_pred, V_tr, obs_traj_batch, r_sum=2 * PED_RADIUS, k=ENERGY_K, tau0=ENERGY_TAU0, tau_max=TAU_MAX):
    mu = V_pred[..., 0:2]

    obs = obs_traj_batch
    while obs.dim() > 3:
        obs = obs.squeeze(0)
    last_obs_pos = obs[:, :, -1].detach()

    abs_pred = torch.cumsum(mu, dim=0) + last_obs_pos.unsqueeze(0)
    abs_gt = torch.cumsum(V_tr, dim=0) + last_obs_pos.unsqueeze(0)
    abs_gt = abs_gt.detach()
    V_tr_vel = V_tr.detach()

    tau = pairwise_ttc_torch_cross(
        abs_pred, mu, abs_gt, V_tr_vel, r_sum, tau_max)
    E = interaction_energy_torch(tau, k, tau0)

    N = mu.shape[1]
    eye = torch.eye(N, dtype=torch.bool, device=mu.device)
    mask = (~eye).float()
    denom = mask.sum().clamp(min=1.0)

    per_t = (torch.tanh(E) * mask).sum(dim=(-2, -1)) / denom
    L_pred_gt = per_t.mean()
    return L_pred_gt


def graph_loss_hybrid(V_pred, V_target, obs_traj_batch, lam, beta_pred_gt=BETA_PRED_GT):
    nll = bivariate_loss(V_pred, V_target)
    if lam > 0:
        l_col = collision_loss(V_pred, obs_traj_batch)
    else:
        l_col = torch.zeros((), device=V_pred.device)

    if beta_pred_gt > 0:
        l_pred_gt = collision_loss_pred_gt(V_pred, V_target, obs_traj_batch)
    else:
        l_pred_gt = torch.zeros((), device=V_pred.device)

    total_loss = nll + lam * l_col + beta_pred_gt * l_pred_gt
    return total_loss, nll.item(), l_col.item(), l_pred_gt.item()


def _scene_pairwise_dist(traj_a, traj_b):
    diff = traj_a[:, :, None, :] - traj_b[:, None, :, :].copy(
    ) if traj_b is not traj_a else traj_a[:, :, None, :] - traj_a[:, None, :, :]
    return np.linalg.norm(diff, axis=-1)


def col1_scene(pred_abs, r_sum=2 * PED_RADIUS):
    N = pred_abs.shape[1]
    dist = _scene_pairwise_dist(pred_abs, pred_abs)
    eye = np.eye(N, dtype=bool)
    dist = np.where(eye[None, :, :], np.inf, dist)
    return 1.0 if np.any(dist <= r_sum) else 0.0


def detect_group_pairs(obs_abs, obs_vel, vel_thresh=GROUP_VEL_THRESH, dist_thresh=GROUP_DIST_THRESH):
    v_ij = obs_vel[:, :, None, :] - obs_vel[:, None, :, :]
    v_ij_mag = np.linalg.norm(v_ij, axis=-1).mean(axis=0)
    x_ij = obs_abs[:, :, None, :] - obs_abs[:, None, :, :]
    x_ij_mag = np.linalg.norm(x_ij, axis=-1).mean(axis=0)
    group_mask = (v_ij_mag < vel_thresh) & (x_ij_mag < dist_thresh)
    np.fill_diagonal(group_mask, False)
    return group_mask


def col1_check_scene(pred_abs, r_sum=2 * PED_RADIUS, group_mask=None):
    N = pred_abs.shape[1]
    dist = _scene_pairwise_dist(pred_abs, pred_abs)
    exclude = np.eye(N, dtype=bool)
    if group_mask is not None:
        exclude = exclude | group_mask
    dist = np.where(exclude[None, :, :], np.inf, dist)
    coll_mask = np.any(dist <= r_sum, axis=(0, 2))
    return coll_mask.astype(float)


def col2_scene(pred_abs, trgt_abs, r_sum=2 * PED_RADIUS):
    N = pred_abs.shape[1]
    diff = pred_abs[:, :, None, :] - trgt_abs[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    eye = np.eye(N, dtype=bool)
    dist = np.where(eye[None, :, :], np.inf, dist)
    return 1.0 if np.any(dist <= r_sum) else 0.0


def col2_rate_scene(pred_abs, trgt_abs, r_sum=2 * PED_RADIUS):
    N = pred_abs.shape[1]
    diff = pred_abs[:, :, None, :] - trgt_abs[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    eye = np.eye(N, dtype=bool)
    return (dist[:, ~eye] <= r_sum).mean()


def col2_check_scene(pred_abs, trgt_abs, r_sum=2 * PED_RADIUS, group_mask=None):
    N = pred_abs.shape[1]
    diff = pred_abs[:, :, None, :] - trgt_abs[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    exclude = np.eye(N, dtype=bool)
    if group_mask is not None:
        exclude = exclude | group_mask
    dist = np.where(exclude[None, :, :], np.inf, dist)
    coll_mask = np.any(dist <= r_sum, axis=(0, 2))
    return coll_mask.astype(float)


def avg_energy_scene(pred_abs, r_sum=2 * PED_RADIUS, k=ENERGY_K, tau0=ENERGY_TAU0, tau_max=TAU_MAX):
    T, N, _ = pred_abs.shape
    vel = np.zeros_like(pred_abs)
    vel[1:] = pred_abs[1:] - pred_abs[:-1]
    eye = np.eye(N, dtype=bool)
    total = 0.0
    for t in range(T):
        tau = pairwise_ttc_np(pred_abs[t], vel[t], r_sum, tau_max)
        E = interaction_energy_np(tau, k, tau0)
        total += E[~eye].sum()
    return float(total)


def min_dist_pairs_pred_pred(pred_abs, group_mask=None):
    N = pred_abs.shape[1]
    dist = _scene_pairwise_dist(pred_abs, pred_abs)
    min_dist = dist.min(axis=0)
    exclude = np.eye(N, dtype=bool)
    if group_mask is not None:
        exclude = exclude | group_mask
    iu = np.triu_indices(N, k=1)
    valid = ~exclude[iu]
    return min_dist[iu][valid].tolist()


def min_dist_pairs_pred_gt(pred_abs, trgt_abs, group_mask=None):
    N = pred_abs.shape[1]
    diff = pred_abs[:, :, None, :] - trgt_abs[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    min_dist = dist.min(axis=0)
    exclude = np.eye(N, dtype=bool)
    if group_mask is not None:
        exclude = exclude | group_mask
    return min_dist[~exclude].tolist()


def summarize_min_dist(vals):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return {k: float('nan') for k in ['mean', 'median', 'p10', 'p25']}
    return {
        'mean': float(v.mean()),
        'median': float(np.median(v)),
        'p10': float(np.percentile(v, 10)),
        'p25': float(np.percentile(v, 25)),
    }


def plot_min_dist_cdf(dist_dict, r_sum=2 * PED_RADIUS, title='CDF min distance', save_path=None):
    plt.figure(figsize=(6, 5))
    for label, vals in dist_dict.items():
        v = np.sort(np.asarray(vals, dtype=float))
        cdf = np.arange(1, len(v) + 1) / len(v)
        plt.plot(v, cdf, label=label)
    plt.axvline(r_sum, color='red', linestyle='--',
                label=f'threshold ({r_sum} m)')
    plt.xlabel('min_t dist (m)')
    plt.ylabel('CDF')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
