import os
import copy
import pickle
import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist
from torch.utils.data import DataLoader

try:
    from .model import social_stgcnn
    from .utils import TrajectoryDataset, TrajectoryDatasetHybrid, PED_RADIUS
    from .metrics import (
        ade, fde, seq_to_nodes, nodes_rel_to_nodes_abs,
        col1_scene, detect_group_pairs, col1_check_scene, col2_scene,
        col2_rate_scene, col2_check_scene, avg_energy_scene,
        min_dist_pairs_pred_pred, min_dist_pairs_pred_gt, summarize_min_dist
    )
except ImportError:
    from model import social_stgcnn
    from utils import TrajectoryDataset, TrajectoryDatasetHybrid, PED_RADIUS
    from metrics import (
        ade, fde, seq_to_nodes, nodes_rel_to_nodes_abs,
        col1_scene, detect_group_pairs, col1_check_scene, col2_scene,
        col2_rate_scene, col2_check_scene, avg_energy_scene,
        min_dist_pairs_pred_pred, min_dist_pairs_pred_gt, summarize_min_dist
    )

def evaluate_model(data_root, dataset_name, ckpt_path, output_dir=None, is_hybrid=True,
                   obs_len=8, pred_len=12, n_stgcnn=1, n_txpcnn=5, output_size=5,
                   kernel_size=3, ksteps=20, r_sum=2 * PED_RADIUS, run_tag="run", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_set = os.path.join(data_root, dataset_name) + '/'
    dataset_cls = TrajectoryDatasetHybrid if is_hybrid else TrajectoryDataset

    dset_test = dataset_cls(
        data_set + 'test/', obs_len=obs_len, pred_len=pred_len,
        skip=1, norm_lap_matr=True)
    loader_test = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=0)

    model = social_stgcnn(n_stgcnn=n_stgcnn, n_txpcnn=n_txpcnn,
                           output_feat=output_size, seq_len=obs_len,
                           kernel_size=kernel_size, pred_seq_len=pred_len).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    mindist_i_ls = []
    mindist_ii_ls = []
    ade_bigls = []
    fde_bigls = []
    col1_ls = []
    col1_check_flat = []
    col2_ls = []
    col2_rate_ls = []
    col2_check_flat = []
    ae_sum_total = 0.0
    ae_M_total = 0
    raw_data_dict = {}
    step = 0

    for batch in loader_test:
        step += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        sx = torch.exp(V_pred[:, :, 2])
        sy = torch.exp(V_pred[:, :, 3])
        corr = torch.tanh(V_pred[:, :, 4])

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2, device=device)
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                  V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                  V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(ksteps):
            V_pred_sample = mvnormal.sample()
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred_sample.data.cpu().numpy().squeeze().copy(),
                                                        V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

        if num_of_objs >= 2:
            mean_abs = nodes_rel_to_nodes_abs(mean.data.cpu().numpy().squeeze().copy(),
                                               V_x[-1, :, :].copy())
            if mean_abs.ndim == 2:
                mean_abs = mean_abs[:, None, :]

            V_obs_np = V_obs.data.cpu().numpy().squeeze().copy()
            group_mask = detect_group_pairs(V_x_rel_to_abs, V_obs_np)

            mindist_i_ls.extend(min_dist_pairs_pred_pred(mean_abs, group_mask=group_mask))
            mindist_ii_ls.extend(min_dist_pairs_pred_gt(mean_abs, V_y_rel_to_abs, group_mask=group_mask))
            col1_ls.append(col1_scene(mean_abs, r_sum))

            coll_mask_i = col1_check_scene(mean_abs, r_sum, group_mask=group_mask)
            col1_check_flat.extend(coll_mask_i.tolist())
            col2_ls.append(col2_scene(mean_abs, V_y_rel_to_abs, r_sum))
            col2_rate_ls.append(col2_rate_scene(mean_abs, V_y_rel_to_abs, r_sum))

            coll_mask_ii = col2_check_scene(mean_abs, V_y_rel_to_abs, r_sum, group_mask=group_mask)
            col2_check_flat.extend(coll_mask_ii.tolist())

            ae_sum_total += avg_energy_scene(mean_abs, r_sum)
            ae_M_total += num_of_objs

    ade_ = sum(ade_bigls) / len(ade_bigls) if len(ade_bigls) > 0 else float('nan')
    fde_ = sum(fde_bigls) / len(fde_bigls) if len(fde_bigls) > 0 else float('nan')
    col1_ = float(np.mean(col1_ls)) if len(col1_ls) > 0 else float('nan')
    col1_check_ = float(np.mean(col1_check_flat)) if len(col1_check_flat) > 0 else float('nan')
    col2_ = float(np.mean(col2_ls)) if len(col2_ls) > 0 else float('nan')
    col2_rate_ = float(np.mean(col2_rate_ls)) if len(col2_rate_ls) > 0 else float('nan')
    col2_check_ = float(np.mean(col2_check_flat)) if len(col2_check_flat) > 0 else float('nan')
    ae_ = float(ae_sum_total / (ae_M_total * pred_len)) if ae_M_total > 0 else float('nan')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'mindist_{run_tag}_{dataset_name}.pkl'), 'wb') as fp:
            pickle.dump({'pred_pred': mindist_i_ls, 'pred_gt': mindist_ii_ls}, fp)

    return {
        'ADE': ade_,
        'FDE': fde_,
        'COL-I': col1_,
        'COL-I-check': col1_check_,
        'COL-II': col2_,
        'COL-II-rate': col2_rate_,
        'COL-II-check': col2_check_,
        'AE': ae_,
        'MinDist-I': summarize_min_dist(mindist_i_ls),
        'MinDist-II': summarize_min_dist(mindist_ii_ls),
        'raw_data_dict': raw_data_dict,
    }

if __name__ == '__main__':
    data_root = "./datasets_eth_hotels_"
    dataset_name = "Leon"
    ckpt_path = "./ttc_stgcnn_runs/Leon_lam0.0/val_best.pth"
    if os.path.exists(ckpt_path):
        res = evaluate_model(data_root, dataset_name, ckpt_path, output_dir="./ttc_stgcnn_runs")
        print("Evaluation results:")
        for k, v in res.items():
            if k != 'raw_data_dict':
                print(f"{k}: {v}")
    else:
        print(f"Checkpoint not found at {ckpt_path}")
