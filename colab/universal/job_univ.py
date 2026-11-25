"""One Universal-Benchmark job: run one model on one dataset under wrapper mode ('raw', 'rad3', 'cone', 'inf')
and data condition ('id', 'r1', 'c0'), evaluating both Input Locality/Accuracy and Output TTC-OSC Sample Coupling.
"""
import argparse, os, sys, time, json, random, hashlib
import numpy as np


def pkeys(obs, gt):
    """64-bit pedestrian-window key from (obs, gt) rounded to 1 mm (invariant to track-ID permutation)."""
    mm = np.round(np.concatenate([obs, gt], 1).reshape(len(obs), -1) * 1000).astype(np.int64)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), 'little', signed=True)
                     for r in mm], dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--ds', required=True)
    p.add_argument('--mode', default='raw', choices=['raw', 'rad3', 'cone', 'inf'])
    p.add_argument('--cond', default='id')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--K', type=int, default=20)
    p.add_argument('--name', help='output model name (e.g. STGCNN-F)')
    p.add_argument('--variant', default='')
    p.add_argument('--ckpt', default='')
    a = p.parse_args()

    if a.variant:
        os.environ['BENCH_VARIANT'] = a.variant
    if a.ckpt:
        os.environ['BENCH_CKPT'] = os.path.abspath(a.ckpt)
    os.environ['BENCH_EGO_MODE'] = '' if a.mode == 'raw' else a.mode

    here = os.path.dirname(os.path.abspath(__file__))
    bench_a = os.path.normpath(os.path.join(here, '..', 'bench_a'))
    sys.path[:0] = [bench_a, here]

    import torch
    import adapters
    from wrap_utils import (detect_groups_np, cone_mask_np, ttc_osc_costs, eval_coupling,
                            assign_seq, solve_ttc_osc, jade_for_sigma, oracle_jade_bounds)

    def seeded(seed=0):
        adapters.SEED = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    mname = a.name or a.model
    tag = f'{a.mode}_{a.cond}'
    out = os.path.abspath(os.path.join(a.out, mname, a.ds))
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    run = adapters.MODELS[a.model][1](a.ds, os.path.abspath(a.root))
    load_s = time.time() - t0

    t1 = time.time()
    seeded(0)
    # Pin noise whenever evaluating ego wrapper or invariance checks so order/bystander tests are strictly deterministic
    use_pin = (a.model in adapters.NEEDS_PIN) and (a.mode != 'raw' or a.cond != 'id')

    sizes, K_all, P_all, O_all = [], [], [], []
    ade_all, fde_all, cen_ade_all = [], [], []
    ade_close, cen_close, ade_app, cen_app = [], [], [], []
    coli_02, coli_04, coli_app = [], [], []

    osc_keys = ('rel', 'rand', 'bin', 'osc1', 'osc2', 'max')
    cols = {k: [] for k in osc_keys}
    col04 = {k: [] for k in ('rel', 'osc2')}
    enrg = {k: [] for k in ('rel', 'osc2')}
    cols_app = {k: [] for k in ('rel', 'osc2')}
    jades = {k: [] for k in ('rel', 'rand', 'osc2', 'omin', 'omax')}
    scene_w, osc_ms = [], []
    rng_shuf = torch.Generator().manual_seed(12345)

    with torch.no_grad():
        for obs, gt, s, mu in run(a.K, use_pin):
            assert s.shape[0] == a.K and s.shape[1:] == gt.shape and obs.shape[1:] == (8, 2), (s.shape, gt.shape, obs.shape)
            n = len(obs)
            obs_np, gt_np = obs.numpy(), gt.numpy()
            pin_t = mu if mu is not None else s.mean(0)
            pin_np = pin_t.numpy().astype(np.float32)

            err = (s - gt).norm(dim=-1)  # K, n, 12
            ade_k = err.mean(-1)         # K, n
            ade_w = ade_k.min(0)[0].numpy().astype(np.float32)
            fde_w = err[..., -1].min(0)[0].numpy().astype(np.float32)
            cen_w = (pin_t - gt).norm(dim=-1).mean(-1).numpy().astype(np.float32)
            ade_all.append(ade_w); fde_all.append(fde_w); cen_ade_all.append(cen_w)

            if n < 2:
                continue

            sizes.append(n)
            K_all.append(pkeys(obs_np, gt_np))
            P_all.append(pin_np)
            if a.mode == 'raw' and a.cond == 'id':
                O_all.append(obs_np.astype(np.float32))

            if a.cond != 'id':
                continue

            gm = detect_groups_np(obs_np)
            excl_np = np.eye(n, dtype=bool) | gm
            excl = torch.from_numpy(excl_np)

            # Ground-truth encounter subgroups (non-group pairs)
            d_obs, _, _, _ = cone_mask_np(obs_np)
            d_fut_gt = np.linalg.norm(gt_np[:, None] - gt_np[None, :], axis=-1).min(-1)
            d_fut_ng = np.where(excl_np, np.inf, d_fut_gt)
            close_m = (d_fut_ng < 1.5).any(-1)
            app_m = ((d_obs > 3.0) & (d_fut_ng < 1.5)).any(-1)
            if close_m.any():
                ade_close.append(ade_w[close_m]); cen_close.append(cen_w[close_m])
            if app_m.any():
                ade_app.append(ade_w[app_m]); cen_app.append(cen_w[app_m])

            # Centroid collision metrics (Col-I)
            d_pred_cen = np.linalg.norm(pin_np[:, None] - pin_np[None, :], axis=-1).min(-1)
            d_pred_cen = np.where(excl_np, np.inf, d_pred_cen)
            ci02 = (d_pred_cen < 0.2).any(-1).astype(np.float32)
            ci04 = (d_pred_cen < 0.4).any(-1).astype(np.float32)
            coli_02.append(ci02); coli_04.append(ci04)
            if app_m.any():
                coli_app.append(ci02[app_m])

            # Output Sample Coupling (TTC-OSC) & JADE bounds
            t_osc0 = time.perf_counter()
            C, H02, H04, Epair = ttc_osc_costs(s, obs[:, -1])
            sig_osc2 = solve_ttc_osc(C, excl, sweeps=2)
            osc_ms.append((time.perf_counter() - t_osc0) * 1000.0)

            sig_rel = torch.arange(a.K, dtype=torch.long).repeat(n, 1)
            sig_osc1 = solve_ttc_osc(C, excl, sweeps=1)
            sig_bin = assign_seq(H02, excl, maximize=False, binary_any=True)
            sig_max = assign_seq(H02, excl, maximize=True, binary_any=True)
            sig_rands = [torch.stack([torch.randperm(a.K, generator=rng_shuf) for _ in range(n)]) for _ in range(5)]

            err_nk = ade_k.T.contiguous()  # n, K
            for kname, sig in (('rel', sig_rel), ('bin', sig_bin), ('osc1', sig_osc1), ('osc2', sig_osc2), ('max', sig_max)):
                c2, c4, ep = eval_coupling(H02, H04, Epair, sig, excl)
                cols[kname].append(c2)
                if kname in col04:
                    col04[kname].append(c4); enrg[kname].append(ep)
                    if app_m.any():
                        cols_app[kname].append(c2[app_m])
                if kname in jades:
                    jades[kname].append(jade_for_sigma(err_nk, sig))

            r_c2 = [eval_coupling(H02, H04, Epair, sr, excl)[0] for sr in sig_rands]
            cols['rand'].append(np.mean(r_c2, axis=0))
            jades['rand'].append(float(np.mean([jade_for_sigma(err_nk, sr) for sr in sig_rands])))
            jmin, jmax = oracle_jade_bounds(err_nk)
            jades['omin'].append(jmin); jades['omax'].append(jmax)
            scene_w.append(n)

    payload = dict(
        sizes=np.array(sizes, dtype=np.int32),
        key=np.concatenate(K_all) if K_all else np.zeros(0, dtype=np.int64),
        pin=np.concatenate(P_all) if P_all else np.zeros((0, 12, 2), dtype=np.float32),
    )
    if O_all:
        payload['obs'] = np.concatenate(O_all)
    np.savez_compressed(os.path.join(out, f'{tag}.npz'), **payload)

    def _m(lst):
        if not lst:
            return None
        arr = np.concatenate(lst) if isinstance(lst[0], np.ndarray) else np.asarray(lst, dtype=np.float64)
        return float(arr.mean()) if len(arr) else None

    def _cnt(lst):
        return int(sum(len(x) for x in lst)) if lst else 0

    w = np.asarray(scene_w, dtype=np.float64)
    meta = dict(
        model=mname, base_model=a.model, variant=a.variant or 'O', ds=a.ds, mode=a.mode, cond=a.cond,
        n_scenes=len(sizes), n_peds=int(sum(sizes)),
        ade_all=_m(ade_all), fde_all=_m(fde_all), cen_ade_all=_m(cen_ade_all),
        n_close=_cnt(ade_close), ade_close=_m(ade_close), cen_close=_m(cen_close),
        n_approach=_cnt(ade_app), ade_approach=_m(ade_app), cen_approach=_m(cen_app),
        coli_02=_m(coli_02), coli_04=_m(coli_04), coli_approach=_m(coli_app),
        load_s=round(load_s, 2), eval_s=round(time.time() - t1, 2),
    )
    if a.cond == 'id' and len(w):
        meta['osc'] = {
            **{f'cols_{k}': _m(cols[k]) for k in osc_keys},
            'col04_rel': _m(col04['rel']), 'col04_osc2': _m(col04['osc2']),
            'energy_rel': _m(enrg['rel']), 'energy_osc2': _m(enrg['osc2']),
            'cols_app_rel': _m(cols_app['rel']), 'cols_app_osc2': _m(cols_app['osc2']),
            **{f'jade_{k}': float(np.average(jades[k], weights=w)) for k in jades},
            'osc2_ms_mean': float(np.mean(osc_ms)),
            'osc2_ms_p95': float(np.percentile(osc_ms, 95)),
        }
    json.dump(meta, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    osc_str = f"cols {meta['osc']['cols_rel']:.4f}->{meta['osc']['cols_osc2']:.4f}" if 'osc' in meta else ''
    print(mname, a.ds, tag, f"ade={meta['ade_all']:.4f} {osc_str} [{meta['eval_s']:.1f}s]", flush=True)


if __name__ == '__main__':
    main()
