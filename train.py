import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

try:
    from .model import social_stgcnn
    from .utils import build_dataloaders, set_seed, PED_RADIUS
    from .metrics import graph_loss_hybrid
    from .test import evaluate_model
except ImportError:
    from model import social_stgcnn
    from utils import build_dataloaders, set_seed, PED_RADIUS
    from metrics import graph_loss_hybrid
    from test import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = "/kaggle/input/datasets/thnhott/dataset-eth-hotel/datasets_eth_hotels_/datasets_eth_hotels_"
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = "./datasets_eth_hotels_"

DATASETS = ["Leon"]
LAMBDAS = [0.0]

OBS_LEN = 8
PRED_LEN = 12
INPUT_SIZE = 2
OUTPUT_SIZE = 5
N_STGCNN = 1
N_TXPCNN = 5
KERNEL_SIZE = 3
NUM_EPOCHS = 250
BATCH_SIZE = 128
LR = 0.01
CLIP_GRAD = None
USE_LRSCHD = True
LR_SH_RATE = 150
BETA_PRED_GT = 1.0

OUTPUT_DIR = "./ttc_stgcnn_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _run_dir(output_dir, dataset_name, lam):
    d = os.path.join(output_dir, f"{dataset_name}_lam{lam}")
    os.makedirs(d, exist_ok=True)
    return d

def train_epoch(model, loader_train, optimizer, lam, beta_pred_gt=BETA_PRED_GT, batch_size=BATCH_SIZE, clip_grad=CLIP_GRAD):
    model.train()
    loss_batch = 0.0
    nll_batch = 0.0
    col_batch = 0.0
    pred_gt_batch = 0.0
    loader_len = len(loader_train)
    optimizer.zero_grad()

    for cnt, batch in enumerate(loader_train):
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        loss, nll_val, col_val, pred_gt_val = graph_loss_hybrid(V_pred, V_tr, obs_traj, lam, beta_pred_gt=beta_pred_gt)
        loss_scaled = loss / batch_size
        loss_scaled.backward()
        loss_batch += loss.item()
        nll_batch += nll_val
        col_batch += col_val
        pred_gt_batch += pred_gt_val

        if (cnt + 1) % batch_size == 0 or (cnt + 1) == loader_len:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad()

    return loss_batch / loader_len, nll_batch / loader_len, col_batch / loader_len, pred_gt_batch / loader_len

def validate_epoch(model, loader_val, lam, beta_pred_gt=BETA_PRED_GT):
    model.eval()
    loss_batch = 0.0
    loader_len = len(loader_val)
    with torch.no_grad():
        for cnt, batch in enumerate(loader_val):
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
            V_pred = V_pred.permute(0, 2, 3, 1)

            V_tr = V_tr.squeeze()
            V_pred = V_pred.squeeze()

            loss, _, _, _ = graph_loss_hybrid(V_pred, V_tr, obs_traj, lam, beta_pred_gt=beta_pred_gt)
            loss_batch += loss.item()

    return loss_batch / loader_len

def train_model(data_root, dataset_name, lam, output_dir=OUTPUT_DIR, num_epochs=NUM_EPOCHS,
                obs_len=OBS_LEN, pred_len=PRED_LEN, is_hybrid=True, verbose=True):
    run_dir = _run_dir(output_dir, dataset_name, lam)
    ckpt_path = os.path.join(run_dir, 'val_best.pth')

    print(f"\n{'='*70}\nDataset={dataset_name}  lambda={lam}  epochs={num_epochs}\n{'='*70}")

    loader_train, loader_val = build_dataloaders(data_root, dataset_name, obs_len=obs_len, pred_len=pred_len, is_hybrid=is_hybrid)

    model = social_stgcnn(n_stgcnn=N_STGCNN, n_txpcnn=N_TXPCNN,
                           output_feat=OUTPUT_SIZE, seq_len=obs_len,
                           kernel_size=KERNEL_SIZE, pred_seq_len=pred_len).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LR)
    scheduler = None
    if USE_LRSCHD:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SH_RATE, gamma=0.2)

    history = {'train_loss': [], 'train_nll': [], 'train_col': [], 'train_pred_gt': [], 'val_loss': []}
    best_val = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        t0 = time.time()
        tr_loss, tr_nll, tr_col, tr_pred_gt = train_epoch(model, loader_train, optimizer, lam)
        val_loss = validate_epoch(model, loader_val, lam)
        epoch_time = time.time() - t0
        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_nll'].append(tr_nll)
        history['train_col'].append(tr_col)
        history['train_pred_gt'].append(tr_pred_gt)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        pct = (epoch + 1) / num_epochs * 100
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"  epoch {epoch:3d}/{num_epochs} [{pct:5.1f}%] | "
                  f"train_loss {tr_loss:.4f} (nll {tr_nll:.4f} + lam*col {lam*tr_col:.4f} + beta*pred_gt {BETA_PRED_GT*tr_pred_gt:.4f}) | "
                  f"val_loss {val_loss:.4f} | best_val {best_val:.4f}@{best_epoch} | "
                  f"time {epoch_time:.1f}s")

    with open(os.path.join(run_dir, 'history.pkl'), 'wb') as fp:
        pickle.dump(history, fp)

    return ckpt_path, history

def main():
    set_seed(42)
    force_retrain = False
    results = []

    for dataset_name in DATASETS:
        for lam in LAMBDAS:
            run_dir = _run_dir(OUTPUT_DIR, dataset_name, lam)
            ckpt_path = os.path.join(run_dir, 'val_best.pth')

            if os.path.exists(ckpt_path) and not force_retrain:
                print(f"[skip-train] {dataset_name} lambda={lam}: checkpoint exists, reusing.")
            else:
                ckpt_path, history = train_model(DATA_ROOT, dataset_name, lam, output_dir=OUTPUT_DIR, num_epochs=NUM_EPOCHS, is_hybrid=True)

            metrics_out = evaluate_model(DATA_ROOT, dataset_name, ckpt_path, output_dir=OUTPUT_DIR, is_hybrid=True, ksteps=20, run_tag="A_moi")
            row = {
                'dataset': dataset_name,
                'lambda': lam,
                'ADE': metrics_out['ADE'],
                'FDE': metrics_out['FDE'],
                'COL-I': metrics_out['COL-I'],
                'COL-I-check': metrics_out['COL-I-check'],
                'COL-II': metrics_out['COL-II'],
                'COL-II-rate': metrics_out['COL-II-rate'],
                'COL-II-check': metrics_out['COL-II-check'],
                'AE': metrics_out['AE']
            }
            results.append(row)
            print(f"[done] {row}")

            with open(os.path.join(OUTPUT_DIR, 'results_partial.pkl'), 'wb') as fp:
                pickle.dump(results, fp)

    print("\nCompleted all", len(results), "(dataset, lambda) runs.")

    if results:
        df_results = pd.DataFrame(results)
        cols = ['dataset', 'lambda', 'ADE', 'FDE', 'COL-I', 'COL-I-check', 'COL-II', 'COL-II-rate', 'COL-II-check', 'AE']
        valid_cols = [c for c in cols if c in df_results.columns]
        df_results = df_results[valid_cols]
        df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
        print("\nResults table:")
        print(df_results)

        metrics_to_plot = [m for m in ['ADE', 'FDE', 'COL-I', 'COL-II', 'COL-II-rate', 'COL-II-check', 'AE'] if m in df_results.columns]
        if len(df_results['lambda'].unique()) > 1:
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.2 * len(metrics_to_plot), 4))
            if len(metrics_to_plot) == 1:
                axes = [axes]
            for ax, m in zip(axes, metrics_to_plot):
                for dataset_name in DATASETS:
                    sub = df_results[df_results['dataset'] == dataset_name].sort_values('lambda')
                    ax.plot(sub['lambda'], sub[m], marker='o', label=dataset_name)
                ax.set_xlabel('lambda')
                ax.set_ylabel(m)
                ax.set_title(m + ' vs lambda')
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_vs_lambda_per_dataset.png'), dpi=150)
            plt.close()

            df_avg = df_results.groupby('lambda')[metrics_to_plot].mean().reset_index()
            df_avg.to_csv(os.path.join(OUTPUT_DIR, 'results_avg_over_datasets.csv'), index=False)

            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.2 * len(metrics_to_plot), 4))
            if len(metrics_to_plot) == 1:
                axes = [axes]
            for ax, m in zip(axes, metrics_to_plot):
                ax.plot(df_avg['lambda'], df_avg[m], marker='o', color='tab:blue')
                ax.set_xlabel('lambda')
                ax.set_ylabel(m)
                ax.set_title(m + ' (average)')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_vs_lambda_avg.png'), dpi=150)
            plt.close()

if __name__ == '__main__':
    main()
