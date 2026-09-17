import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PED_RADIUS = 0.2
dt = 0.4
ENERGY_K = 1.5
ENERGY_TAU0 = 3.0
TAU_MAX = 12.0
EPS_ENERGY = 0.01
ALPHA_HYBRID = 1.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / NORM


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    if seq_rel.ndim == 2:
        seq_rel = seq_rel[np.newaxis, :, :]
    seq_len = seq_rel.shape[2]
    max_nodes = seq_rel.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_rel = seq_rel[:, :, s]
        V[s, :, :] = step_rel
        diff = step_rel[:, None, :] - step_rel[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, 1.0)
        A[s, :, :] = dist
        if norm_lap_matr:
            d = dist.sum(axis=1)
            d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
            D_inv = np.diag(d_inv_sqrt)
            A[s, :, :] = np.eye(max_nodes) - D_inv @ dist @ D_inv

    return torch.from_numpy(V).type(torch.float), torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim=None):
    data = []
    with open(_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split()
            items = [float(i) for i in items]
            data.append(items)
    return np.asarray(data)


def pairwise_ttc_np(pos, vel, r_sum, tau_max=TAU_MAX, eps=1e-6):
    N = pos.shape[0]
    x_ij = pos[:, None, :] - pos[None, :, :]
    v_ij = vel[:, None, :] - vel[None, :, :]

    v_sq = np.sum(v_ij ** 2, axis=-1)
    x_sq = np.sum(x_ij ** 2, axis=-1)
    x_dot_v = np.sum(x_ij * v_ij, axis=-1)

    disc = x_dot_v ** 2 - v_sq * (x_sq - r_sum ** 2)

    tau = np.full((N, N), tau_max, dtype=np.float64)

    moving = v_sq > eps
    real_roots = disc >= 0
    safe_vsq = np.where(moving, v_sq, 1.0)
    sqrt_disc = np.sqrt(np.clip(disc, 0, None))
    candidate = (-x_dot_v - sqrt_disc) / safe_vsq

    valid_future_collision = moving & real_roots & (candidate > 0)
    tau = np.where(valid_future_collision, np.minimum(candidate, tau_max), tau)

    already_colliding = x_sq <= (r_sum ** 2)
    tau = np.where(already_colliding, 0.0, tau)

    np.fill_diagonal(tau, tau_max)
    return tau


def interaction_energy_np(tau, k=ENERGY_K, tau0=ENERGY_TAU0, eps=EPS_ENERGY):
    tau_sec = tau * dt
    E = (k / (tau_sec ** 2 + eps)) * np.exp(-tau_sec / tau0)
    return E


def seq_to_graph_hybrid(seq_, seq_rel, alpha=ALPHA_HYBRID, r_sum=2 * PED_RADIUS, norm_lap_matr=True):
    seq_abs = seq_.squeeze()
    seq_r = seq_rel.squeeze()

    if hasattr(seq_abs, 'numpy'):
        seq_abs = seq_abs.cpu().numpy()
    if hasattr(seq_r, 'numpy'):
        seq_r = seq_r.cpu().numpy()

    if seq_abs.ndim == 2:
        seq_abs = seq_abs[np.newaxis, :, :]
    if seq_r.ndim == 2:
        seq_r = seq_r[np.newaxis, :, :]

    seq_len = seq_r.shape[2]
    max_nodes = seq_r.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    E_all = np.zeros((seq_len, max_nodes, max_nodes))

    for s in range(seq_len):
        step_rel = seq_r[:, :, s]
        step_abs = seq_abs[:, :, s]
        V[s, :, :] = step_rel
        tau = pairwise_ttc_np(step_abs, step_rel, r_sum)
        E = interaction_energy_np(tau)
        E_all[s, :, :] = E
        A_hat = np.tanh(E) + np.eye(max_nodes)
        d = A_hat.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv = np.diag(d_inv_sqrt)
        A_base = D_inv @ A_hat @ D_inv
        A[s, :, :] = A_base

    if np.isnan(A).any() or np.isinf(A).any():
        A = np.nan_to_num(A)

    return torch.from_numpy(V).type(torch.float), torch.from_numpy(A).type(torch.float), torch.from_numpy(E_all).type(torch.float)


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1, delim='\t', norm_lap_matr=True):
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path)
                     for _path in all_files if _path.endswith('.txt')]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(
                    self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros(
                    (len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros(
                    (len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1]
                                                 == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if (pad_end - pad_front != self.seq_len) or (curr_ped_seq.shape[0] != self.seq_len):
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:,
                                                           1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(
            non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            v_, a_ = seq_to_graph(
                self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(
                self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out


class TrajectoryDatasetHybrid(TrajectoryDataset):
    def __init__(self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1, delim='\t', norm_lap_matr=True, alpha=ALPHA_HYBRID, r_sum=2 * PED_RADIUS):
        super().__init__(data_dir, obs_len, pred_len, skip,
                         threshold, min_ped, delim, norm_lap_matr)
        self.E_obs = []
        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            v_, a_, e_ = seq_to_graph_hybrid(
                self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :],
                alpha=alpha, r_sum=r_sum, norm_lap_matr=norm_lap_matr)
            self.v_obs[ss] = v_.clone()
            self.A_obs[ss] = a_.clone()
            self.E_obs.append(e_.clone())


def build_dataloaders(data_root, dataset_name, obs_len=8, pred_len=12, is_hybrid=True):
    data_set = os.path.join(data_root, dataset_name) + '/'
    dataset_cls = TrajectoryDatasetHybrid if is_hybrid else TrajectoryDataset

    dset_train = dataset_cls(
        data_set + 'train/', obs_len=obs_len, pred_len=pred_len,
        skip=1, norm_lap_matr=True)
    loader_train = DataLoader(dset_train, batch_size=1,
                              shuffle=True, num_workers=0)

    dset_val = dataset_cls(
        data_set + 'val/', obs_len=obs_len, pred_len=pred_len,
        skip=1, norm_lap_matr=True)
    loader_val = DataLoader(dset_val, batch_size=1,
                            shuffle=False, num_workers=0)

    return loader_train, loader_val
