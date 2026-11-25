"""Unified post-training evaluator & selftest for IJCAI 2027 Kaggle GPU kernels:
1. Locality & Kinematic TTC-Cone Audit (0-1..>10m, max_gt3, max_gt6, in_cone, out_cone_max, de-confounded far by_dv).
2. Subgroup Anticipation Accuracy (overall vs. close <1.5m vs. long-range approaching d_obs>3m & d_fut<1.5m, Col-I).
3. Realistic Multi-Object Tracking Perturbation Suite (MOT-Glitch: ID-switch, peripheral FOV drop, peripheral ghost spawn).
4. Exhaustive TTC-OSC (1/2/3-sweep BCD Hungarian) & JADE Oracle-vs-Blind Dichotomy (ETH/UCY + SDD).
"""
import os, sys, glob, json, time, argparse
import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'src'), os.path.join(ROOT, 'colab', 'universal'), ROOT]

from common import load_split, scene_graph, forward as single_forward  # noqa: E402
from fast import make_fast_model, batched_forward  # noqa: E402
from train_fast import order_shift  # noqa: E402
from wrap_utils import (cone_mask_np, detect_groups_np, ttc_osc_costs, eval_coupling,  # noqa: E402
                        assign_seq, solve_ttc_osc, jade_for_sigma, oracle_jade_bounds)

D_BINS = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
V_BINS = [0.0, 0.05, 0.1, 0.2, 0.4, np.inf]


def discover_runs(runs_dir, ds):
    """Return sorted list of (cfg, seed, run_dir) for dataset ds."""
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, f'{ds}_IJCAI-*_s*'))):
        if os.path.exists(os.path.join(d, 'val_best.pth')) and os.path.exists(os.path.join(d, 'result.json')):
            base = os.path.basename(d)
            mid = base.split('_IJCAI-', 1)[1]
            cfg, s_str = mid.rsplit('_s', 1)
            out.append((cfg, int(s_str), d))
    return sorted(out)


def load_trained(d, dev):
    c = json.load(open(os.path.join(d, 'result.json')))
    m = make_fast_model(arch=c.get('arch', 'stgcnn'), txp_kernel=c.get('txp', [3, 1]), pool=c.get('pool', False))
    m.load_state_dict(torch.load(os.path.join(d, 'val_best.pth'), map_location='cpu'))
    return m.to(dev).eval(), c


def lean_scenes(scenes, dev):
    return [{'V_obs': s['V_obs'].to(dev), 'V_pred': s['V_pred'].to(dev), 'obs': s['obs'].to(dev)} for s in scenes]


@torch.no_grad()
def final_pos_lean(model, scenes, graph, bs=128):
    out = []
    for b in range(0, len(scenes), bs):
        chunk = scenes[b:b + bs]
        Vp, _ = batched_forward(model, chunk, graph)
        disp = Vp[..., :2].sum(0).cpu()
        p = 0
        for s in chunk:
            n = s['V_obs'].shape[1]
            out.append(s['obs'][:, :, -1].cpu() + disp[p:p + n])
            p += n
    return out


def build_removal_lean(te, max_rm=8, max_scenes=600):
    base, subs, meta = [], [], []
    cand_sc = [s for s in te if s['V_obs'].shape[1] >= 3]
    if len(cand_sc) > max_scenes:
        pick = sorted(np.random.RandomState(42).choice(len(cand_sc), max_scenes, replace=False).tolist())
        cand_sc = [cand_sc[i] for i in pick]
    for si, s in enumerate(cand_sc):
        N = s['V_obs'].shape[1]
        bi = len(base)
        base.append({'V_obs': s['V_obs'], 'V_pred': s['V_pred'], 'obs': s['obs']})
        obs_w = s['obs'].permute(0, 2, 1).cpu().numpy()  # N, 8, 2
        d_obs, d_cpa, _, in_cone = cone_mask_np(obs_w)
        dv = (s['V_obs'][:, :, None] - s['V_obs'][:, None]).norm(dim=-1).mean(0).cpu().numpy()
        js = np.random.RandomState(1000 + si).choice(N, min(N, max_rm), replace=False)
        for j in js:
            keep = [i for i in range(N) if i != j]
            subs.append({'V_obs': s['V_obs'][:, keep], 'V_pred': s['V_pred'][:, keep], 'obs': s['obs'][keep]})
            meta.append((bi, keep, d_obs[keep, j], dv[keep, j], in_cone[keep, j]))
    return base, subs, meta


def binned(x, key, bins):
    idx = np.digitize(key, bins) - 1
    return [dict(lo=float(bins[b]), hi=float(bins[b + 1]), n=int((idx == b).sum()),
                 mean=float(x[idx == b].mean()) if (idx == b).any() else None) for b in range(len(bins) - 1)]


def eval_locality(model, cfg_dict, base_dev, subs_dev, meta):
    g = cfg_dict['graph']
    pb = final_pos_lean(model, base_dev, g)
    ps = final_pos_lean(model, subs_dev, g)
    ch, dd, vv, ic = [], [], [], []
    for (bi, keep, dist, dv, incone), p in zip(meta, ps):
        ch.append((p - pb[bi][keep]).norm(dim=-1).numpy())
        dd.append(dist); vv.append(dv); ic.append(incone)
    if not ch:
        return {}
    ch, dd, vv, ic = np.concatenate(ch), np.concatenate(dd), np.concatenate(vv), np.concatenate(ic)
    far_m = dd > 5.0
    out_m = ~ic
    near_v = float(ch[dd < 2.0].mean()) if (dd < 2.0).any() else None
    far_v = float(ch[far_m].mean()) if far_m.any() else None
    return dict(
        n_pairs=int(len(ch)),
        mean=float(ch.mean()),
        near=near_v,
        far=far_v,
        far_near_ratio=float(far_v / near_v) if (near_v and near_v > 1e-6 and far_v is not None) else None,
        max_gt3=float(ch[dd > 3.0].max()) if (dd > 3.0).any() else 0.0,
        max_gt6=float(ch[dd > 6.0].max()) if (dd > 6.0).any() else 0.0,
        in_cone_mean=float(ch[ic].mean()) if ic.any() else None,
        out_cone_mean=float(ch[out_m].mean()) if out_m.any() else 0.0,
        out_cone_max=float(ch[out_m].max()) if out_m.any() else 0.0,
        by_dist=binned(ch, dd, D_BINS),
        by_dv_far=binned(ch[far_m], vv[far_m], V_BINS) if far_m.any() else [],
    )


@torch.no_grad()
def eval_mot_glitches(model, cfg_dict, te_cpu, dev, max_scenes=350):
    """Evaluate ego endpoint jitter (m) under 3 realistic MOT tracker glitches:
    1. id_switch  : swap 2 pedestrian track slots in the scene
    2. fov_drop   : drop 1 peripheral pedestrian outside causal cone (d_obs > 5m & d_cpa > 3m)
    3. ghost_fp   : spawn 1 false-positive ghost track 8m away moving parallel to ego 0
    """
    g = cfg_dict['graph']
    cand = [s for s in te_cpu if s['V_obs'].shape[1] >= 3]
    if len(cand) > max_scenes:
        pick = sorted(np.random.RandomState(7).choice(len(cand), max_scenes, replace=False).tolist())
        cand = [cand[i] for i in pick]
    if not cand:
        return {}

    base_dev = lean_scenes(cand, dev)
    pb = final_pos_lean(model, base_dev, g)

    sw_scenes, sw_meta = [], []
    drop_scenes, drop_meta = [], []
    gh_scenes, gh_meta = [], []
    rng = np.random.RandomState(99)

    for bi, s in enumerate(cand):
        N = s['V_obs'].shape[1]
        # 1. ID switch: swap two distinct indices
        i0, i1 = rng.choice(N, 2, replace=False)
        pm = list(range(N)); pm[i0], pm[i1] = pm[i1], pm[i0]
        pm_t = torch.tensor(pm, dtype=torch.long)
        sw_scenes.append({'V_obs': s['V_obs'][:, pm_t].to(dev), 'V_pred': s['V_pred'][:, pm_t].to(dev), 'obs': s['obs'][pm_t].to(dev)})
        sw_meta.append((bi, pm))

        # 2. Peripheral FOV drop: find any pair (ego, rem) with d_obs > 5m & d_cpa > 3m
        obs_w = s['obs'].permute(0, 2, 1).numpy()
        d_obs, d_cpa, _, _ = cone_mask_np(obs_w)
        periph = [j for j in range(N) if ((d_obs[:, j] > 5.0) & (d_cpa[:, j] > 3.0)).any()]
        if periph:
            rj = int(periph[0])
            keep = [i for i in range(N) if i != rj]
            ego_ok = [u for u, orig_i in enumerate(keep) if d_obs[orig_i, rj] > 5.0 and d_cpa[orig_i, rj] > 3.0]
            if ego_ok:
                drop_scenes.append({'V_obs': s['V_obs'][:, keep].to(dev), 'V_pred': s['V_pred'][:, keep].to(dev), 'obs': s['obs'][keep].to(dev)})
                drop_meta.append((bi, keep, ego_ok))

        # 3. Peripheral Ghost FP spawn: offset +[8.0, 8.0] m from ped 0 (d_obs >= 11.3m, parallel -> d_cpa >= 11.3m)
        offset = torch.tensor([[[8.0], [8.0]]], dtype=s['obs'].dtype)  # 1, 2, 1
        obs_g = torch.cat([s['obs'], s['obs'][:1] + offset], dim=0)
        v_g = torch.cat([s['V_obs'], s['V_obs'][:, :1]], dim=1)
        vp_g = torch.cat([s['V_pred'], s['V_pred'][:, :1]], dim=1)
        # Verify which original peds have ghost outside their cone
        gw = obs_g.permute(0, 2, 1).numpy()
        _, _, _, g_cone = cone_mask_np(gw)
        far_orig = [i for i in range(N) if not g_cone[i, N]]
        if far_orig:
            gh_scenes.append({'V_obs': v_g.to(dev), 'V_pred': vp_g.to(dev), 'obs': obs_g.to(dev)})
            gh_meta.append((bi, N, far_orig))

    p_sw = final_pos_lean(model, sw_scenes, g)
    sw_sh = []
    for (bi, pm), p in zip(sw_meta, p_sw):
        inv = np.argsort(pm)
        sw_sh.extend((p[inv] - pb[bi]).norm(dim=-1).tolist())

    dr_sh = []
    if drop_scenes:
        p_dr = final_pos_lean(model, drop_scenes, g)
        for (bi, keep, ego_ok), p in zip(drop_meta, p_dr):
            orig_idx = [keep[u] for u in ego_ok]
            dr_sh.extend((p[ego_ok] - pb[bi][orig_idx]).norm(dim=-1).tolist())

    gh_sh = []
    if gh_scenes:
        p_gh = final_pos_lean(model, gh_scenes, g)
        for (bi, N, far_orig), p in zip(gh_meta, p_gh):
            gh_sh.extend((p[:N][far_orig] - pb[bi][far_orig]).norm(dim=-1).tolist())

    return dict(
        id_switch_mean=float(np.mean(sw_sh)) if sw_sh else 0.0,
        id_switch_max=float(np.max(sw_sh)) if sw_sh else 0.0,
        fov_drop_mean=float(np.mean(dr_sh)) if dr_sh else None,
        fov_drop_max=float(np.max(dr_sh)) if dr_sh else None,
        ghost_fp_mean=float(np.mean(gh_sh)) if gh_sh else 0.0,
        ghost_fp_max=float(np.max(gh_sh)) if gh_sh else 0.0,
    )


@torch.no_grad()
def eval_subgroups_and_osc(model, cfg_dict, te_cpu, dev, run_osc=False, max_osc_scenes=400):
    """Evaluate subgroup accuracy (overall, close <1.5m, approaching d_obs>3m & d_fut<1.5m), Col-I, and optionally TTC-OSC."""
    g = cfg_dict['graph']
    te_dev = lean_scenes(te_cpu, dev)
    Vp_all, _ = batched_forward(model, te_dev, g)
    Vp_all = Vp_all.cpu()

    torch.manual_seed(0)
    rng_shuf = torch.Generator().manual_seed(12345)
    ade_all, fde_all, ade_close, cen_close, ade_app, cen_app = [], [], [], [], [], []
    coli_02, coli_app = [], []

    osc_keys = ('rel', 'rand', 'bin', 'osc1', 'osc2', 'osc3', 'max')
    cols = {k: [] for k in osc_keys}
    col04_rel, col04_osc2, en_rel, en_osc2 = [], [], [], []
    jades = {k: [] for k in ('omin', 'rel', 'rand', 'osc2', 'omax')}
    w_osc, ms_osc = [], []

    p = 0
    osc_count = 0
    for sc in te_cpu:
        n = sc['V_obs'].shape[1]
        Vp = Vp_all[:, p:p + n]  # T=12, n, 5
        p += n
        sx, sy, corr = torch.exp(Vp[..., 2]), torch.exp(Vp[..., 3]), torch.tanh(Vp[..., 4])
        cov = torch.stack([sx * sx, corr * sx * sy, corr * sx * sy, sy * sy], -1).view(*sx.shape, 2, 2)
        samp = torchdist.MultivariateNormal(Vp[..., :2], cov).sample((20,))  # 20, 12, n, 2
        last = sc['obs'][:, :, -1]  # n, 2
        gt = torch.cumsum(sc['V_pred'], 0) + last          # 12, n, 2
        pa = torch.cumsum(samp, 1) + last                  # 20, 12, n, 2
        mu_a = torch.cumsum(Vp[..., :2], 0) + last         # 12, n, 2

        err = (pa - gt).norm(dim=-1)                       # 20, 12, n
        ade_kn = err.mean(1)                               # 20, n
        ade_w = ade_kn.min(0).values.numpy()
        fde_w = err[:, -1].min(0).values.numpy()
        cen_w = (mu_a - gt).norm(dim=-1).mean(0).numpy()
        ade_all.append(ade_w); fde_all.append(fde_w)

        if n < 2:
            continue

        obs_w = sc['obs'].permute(0, 2, 1).numpy()         # n, 8, 2
        gt_w = gt.permute(1, 0, 2).numpy()                 # n, 12, 2
        mu_w = mu_a.permute(1, 0, 2).numpy()               # n, 12, 2
        gm = detect_groups_np(obs_w)
        excl_np = np.eye(n, dtype=bool) | gm
        excl = torch.from_numpy(excl_np)

        d_obs, _, _, _ = cone_mask_np(obs_w)
        d_fut = np.where(excl_np, np.inf, np.linalg.norm(gt_w[:, None] - gt_w[None, :], axis=-1).min(-1))
        close_m = (d_fut < 1.5).any(-1)
        app_m = ((d_obs > 3.0) & (d_fut < 1.5)).any(-1)
        if close_m.any():
            ade_close.append(ade_w[close_m]); cen_close.append(cen_w[close_m])
        if app_m.any():
            ade_app.append(ade_w[app_m]); cen_app.append(cen_w[app_m])

        d_mu = np.where(excl_np, np.inf, np.linalg.norm(mu_w[:, None] - mu_w[None, :], axis=-1).min(-1))
        ci = (d_mu < 0.2).any(-1).astype(np.float32)
        coli_02.append(ci)
        if app_m.any():
            coli_app.append(ci[app_m])

        if run_osc and osc_count < max_osc_scenes:
            osc_count += 1
            s_knt2 = pa.permute(0, 2, 1, 3).contiguous()   # 20, n, 12, 2
            t0 = time.perf_counter()
            C, H02, H04, Epair = ttc_osc_costs(s_knt2, last)
            sig_osc2 = solve_ttc_osc(C, excl, sweeps=2)
            ms_osc.append((time.perf_counter() - t0) * 1000.0)

            sig_rel = torch.arange(20, dtype=torch.long).repeat(n, 1)
            sig_osc1 = solve_ttc_osc(C, excl, sweeps=1)
            sig_osc3 = solve_ttc_osc(C, excl, sweeps=3)
            sig_bin = assign_seq(H02, excl, maximize=False, binary_any=True)
            sig_max = assign_seq(H02, excl, maximize=True, binary_any=True)
            sig_rands = [torch.stack([torch.randperm(20, generator=rng_shuf) for _ in range(n)]) for _ in range(5)]

            err_nk = ade_kn.T.contiguous()
            for kname, sig in (('rel', sig_rel), ('bin', sig_bin), ('osc1', sig_osc1),
                               ('osc2', sig_osc2), ('osc3', sig_osc3), ('max', sig_max)):
                c2, c4, ep = eval_coupling(H02, H04, Epair, sig, excl)
                cols[kname].append(c2)
                if kname == 'rel':
                    col04_rel.append(c4); en_rel.append(ep)
                elif kname == 'osc2':
                    col04_osc2.append(c4); en_osc2.append(ep)
                if kname in jades:
                    jades[kname].append(jade_for_sigma(err_nk, sig))
            cols['rand'].append(np.mean([eval_coupling(H02, H04, Epair, sr, excl)[0] for sr in sig_rands], axis=0))
            jades['rand'].append(float(np.mean([jade_for_sigma(err_nk, sr) for sr in sig_rands])))
            jmin, jmax = oracle_jade_bounds(err_nk)
            jades['omin'].append(jmin); jades['omax'].append(jmax)
            w_osc.append(n)

    def _m(lst):
        if not lst:
            return None
        a = np.concatenate(lst) if isinstance(lst[0], np.ndarray) else np.asarray(lst, dtype=np.float64)
        return float(a.mean()) if len(a) else None

    out = dict(
        ADE=_m(ade_all), FDE=_m(fde_all),
        ade_close=_m(ade_close), cen_close=_m(cen_close),
        ade_approach=_m(ade_app), cen_approach=_m(cen_app),
        coli_02=_m(coli_02), coli_approach=_m(coli_app),
    )
    if run_osc and w_osc:
        w = np.asarray(w_osc, dtype=np.float64)
        out['osc'] = {
            **{f'cols_{k}': _m(cols[k]) for k in osc_keys},
            'col04_rel': _m(col04_rel), 'col04_osc2': _m(col04_osc2),
            'energy_rel': _m(en_rel), 'energy_osc2': _m(en_osc2),
            **{f'jade_{k}': float(np.average(jades[k], weights=w)) for k in jades},
            'osc2_ms': float(np.mean(ms_osc)),
        }
    return out


def fmt(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def run_all(runs_dir, datasets, out_prefix):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary = {'device': str(dev), 'records': [], 'by_cfg_eth5': {}, 'by_cfg_sdd': {}}
    osc_cfgs = {'B', 'K', 'K0', 'KN', 'KC', 'TFG', 'TFW', 'TFR1', 'TFC1'}

    for ds in datasets:
        found = discover_runs(runs_dir, ds)
        if not found:
            continue
        try:
            te_cpu = load_split(ds, 'test', need=())
        except Exception as e:
            print(f'skip {ds}: {e}', flush=True)
            continue
        base_l, subs_l, meta_l = build_removal_lean(te_cpu, max_rm=8, max_scenes=500)
        base_dev, subs_dev = lean_scenes(base_l, dev), lean_scenes(subs_l, dev)
        print(f'[{ds}] {len(found)} trained runs, {len(base_l)} loc scenes, {len(subs_l)} removals on {dev}', flush=True)

        for cfg, seed, rdir in found:
            t0 = time.time()
            model, cdict = load_trained(rdir, dev)
            loc = eval_locality(model, cdict, base_dev, subs_dev, meta_l)
            do_osc = (cfg in osc_cfgs) and (seed == 0 or (ds != 'sdd' and seed < 3))
            sub = eval_subgroups_and_osc(model, cdict, te_cpu, dev, run_osc=do_osc)
            mot = eval_mot_glitches(model, cdict, te_cpu, dev) if seed < 3 else {}
            model_cpu = model.cpu()
            oshift = order_shift(model_cpu, te_cpu[:300], cdict['graph'])
            rec = dict(ds=ds, cfg=cfg, seed=seed, arch=cdict.get('arch', 'stgcnn'), graph=cdict['graph'],
                       order_shift=oshift, **sub, locality=loc, mot=mot)
            summary['records'].append(rec)
            print(f"  {ds} {cfg} s{seed}: ADE={fmt(rec['ADE'],3)} app={fmt(rec['ade_approach'],3)} "
                  f"near={fmt(loc.get('near'),4)} outConeMax={fmt(loc.get('out_cone_max'),6)} "
                  f"ord={oshift:.5f} [{time.time()-t0:.1f}s]", flush=True)

    # Aggregate 5-split ETH/UCY and SDD summaries by cfg
    eth_ds = [d for d in ('eth', 'hotel', 'univ', 'zara1', 'zara2') if any(r['ds'] == d for r in summary['records'])]
    cfgs_seen = sorted({r['cfg'] for r in summary['records']})

    def _agg(recs):
        def _mn(fn):
            vals = [fn(r) for r in recs]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        def _mx(fn):
            vals = [fn(r) for r in recs]
            vals = [v for v in vals if v is not None]
            return float(np.max(vals)) if vals else None

        osc_r = [r['osc'] for r in recs if 'osc' in r]
        mot_r = [r['mot'] for r in recs if r.get('mot')]
        return dict(
            n_runs=len(recs),
            ADE=_mn(lambda r: r['ADE']),
            FDE=_mn(lambda r: r['FDE']),
            ade_approach=_mn(lambda r: r['ade_approach']),
            cen_approach=_mn(lambda r: r['cen_approach']),
            coli_02=_mn(lambda r: r['coli_02']),
            coli_approach=_mn(lambda r: r['coli_approach']),
            order_shift=_mn(lambda r: r['order_shift']),
            near=_mn(lambda r: r['locality'].get('near')),
            far=_mn(lambda r: r['locality'].get('far')),
            max_gt3=_mx(lambda r: r['locality'].get('max_gt3')),
            max_gt6=_mx(lambda r: r['locality'].get('max_gt6')),
            in_cone_mean=_mn(lambda r: r['locality'].get('in_cone_mean')),
            out_cone_mean=_mn(lambda r: r['locality'].get('out_cone_mean')),
            out_cone_max=_mx(lambda r: r['locality'].get('out_cone_max')),
            mot_id_switch=float(np.mean([m['id_switch_mean'] for m in mot_r])) if mot_r else None,
            mot_fov_drop=float(np.mean([m['fov_drop_mean'] for m in mot_r if m['fov_drop_mean'] is not None])) if any(m['fov_drop_mean'] is not None for m in mot_r) else None,
            mot_ghost_fp=float(np.mean([m['ghost_fp_mean'] for m in mot_r])) if mot_r else None,
            cols_rel=float(np.mean([o['cols_rel'] for o in osc_r])) if osc_r else None,
            cols_osc1=float(np.mean([o['cols_osc1'] for o in osc_r])) if osc_r else None,
            cols_osc2=float(np.mean([o['cols_osc2'] for o in osc_r])) if osc_r else None,
            cols_osc3=float(np.mean([o['cols_osc3'] for o in osc_r])) if osc_r else None,
            cols_max=float(np.mean([o['cols_max'] for o in osc_r])) if osc_r else None,
            jade_omin=float(np.mean([o['jade_omin'] for o in osc_r])) if osc_r else None,
            jade_rel=float(np.mean([o['jade_rel'] for o in osc_r])) if osc_r else None,
            jade_osc2=float(np.mean([o['jade_osc2'] for o in osc_r])) if osc_r else None,
            jade_omax=float(np.mean([o['jade_omax'] for o in osc_r])) if osc_r else None,
        )

    for cfg in cfgs_seen:
        er = [r for r in summary['records'] if r['cfg'] == cfg and r['ds'] in eth_ds]
        if er:
            # Equal-weight across ETH/UCY datasets
            per_d = [_agg([r for r in er if r['ds'] == d]) for d in eth_ds if any(r['ds'] == d for r in er)]
            summary['by_cfg_eth5'][cfg] = {k: (sum(x[k] for x in per_d) if k == 'n_runs'
                                               else max([x[k] for x in per_d if x[k] is not None], default=None) if k.startswith(('max_', 'out_cone_max'))
                                               else float(np.mean([x[k] for x in per_d if x[k] is not None])) if any(x[k] is not None for x in per_d) else None)
                                           for k in per_d[0]}
        sr = [r for r in summary['records'] if r['cfg'] == cfg and r['ds'] == 'sdd']
        if sr:
            summary['by_cfg_sdd'][cfg] = _agg(sr)

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    json.dump(summary, open(out_prefix + '.json', 'w', encoding='utf-8'), indent=1)

    md = ['# IJCAI 2027 Constructive Architecture, MOT-Glitch & TTC-OSC Summary\n']
    for title, bucket in (('ETH/UCY (Equal-Weight Across Splits)', summary['by_cfg_eth5']),
                          ('Stanford Drone Dataset (SDD)', summary['by_cfg_sdd'])):
        if not bucket:
            continue
        md.append(f'## {title}\n')
        md.append('| Config | Runs | ADE | Appr. ADE | Col-I | Order Shift | Near `<2m` | Far `>5m` | Max `>3m` | Max `>6m` | Out-Cone Max | MOT Ghost | Col-S Rel→OSC2 |')
        md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
        for cfg, st in bucket.items():
            cs = f"{fmt(st['cols_rel'],4)}→{fmt(st['cols_osc2'],4)}" if st['cols_rel'] is not None else '—'
            md.append(f"| {cfg} | {st['n_runs']} | {fmt(st['ADE'],3)} | {fmt(st['ade_approach'],3)} | "
                      f"{fmt(st['coli_02'],4)} | {fmt(st['order_shift'],5)} | {fmt(st['near'],4)} | {fmt(st['far'],4)} | "
                      f"{fmt(st['max_gt3'],5)} | {fmt(st['max_gt6'],5)} | **{fmt(st['out_cone_max'],6)}** | "
                      f"{fmt(st['mot_ghost_fp'],5)} | {cs} |")
        md.append('')

    open(out_prefix + '.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print(f'Saved {out_prefix}.json and {out_prefix}.md', flush=True)


def selftest():
    torch.manual_seed(0)
    # Build synthetic scene with:
    # ped 0 at origin moving +x; ped 1 at 1.8m; ped 2 head-on at x=6.0m closing fast (inside TTC cone, >3m!);
    # ped 3 at (0, 4.5m) (between 3m and 6m, linked to ped 0 via ped 4 at (0, 2.2m) for 2-hop test!);
    # ped 5 at (20m, 20m) (strictly outside 6m and outside TTC cone!).
    T = 8
    N = 6
    obs = torch.zeros(N, 2, T)
    ts = torch.arange(T, dtype=torch.float32)
    obs[0, 0] = 0.3 * ts                       # at t=7: (2.1, 0)
    obs[1, 0] = 0.3 * ts; obs[1, 1] = 1.5      # 1.5m from ped 0
    obs[2, 0] = 8.5 - 0.4 * ts                 # at t=7: (5.7, 0), d_obs=3.6m > 3m, closing at 0.7m/step -> d_cpa=0.0m!
    obs[3, 0] = 2.1; obs[3, 1] = 4.6           # 4.6m from ped 0, 2.2m from ped 4 -> 2-hop reachable!
    obs[4, 0] = 2.1; obs[4, 1] = 2.4           # 2.4m from ped 0 and 2.2m from ped 3
    obs[5, 0] = 25.0; obs[5, 1] = 25.0         # far bystander (>20m)

    v_obs = torch.zeros(T, N, 2)
    v_obs[1:] = (obs[:, :, 1:] - obs[:, :, :-1]).permute(2, 0, 1)
    v_pred = torch.randn(12, N, 2) * 0.1
    sc_full = {'obs': obs, 'V_obs': v_obs, 'V_pred': v_pred}
    keep_no5 = [0, 1, 2, 3, 4]
    keep_no2 = [0, 1, 3, 4, 5]
    keep_no3 = [0, 1, 2, 4, 5]
    sc_no5 = {'obs': obs[keep_no5], 'V_obs': v_obs[:, keep_no5], 'V_pred': v_pred[:, keep_no5]}
    sc_no2 = {'obs': obs[keep_no2], 'V_obs': v_obs[:, keep_no2], 'V_pred': v_pred[:, keep_no2]}
    sc_no3 = {'obs': obs[keep_no3], 'V_obs': v_obs[:, keep_no3], 'V_pred': v_pred[:, keep_no3]}

    # 1. Verify GCN A_cone: exact 0 shift on removing ped 5 (outside cone), NON-ZERO shift on removing head-on ped 2 (>3m)!
    m_kc = make_fast_model('stgcnn', (3, 1)).eval()
    p_full = final_pos_lean(m_kc, [sc_full], 'A_cone')[0]
    p_no5 = final_pos_lean(m_kc, [sc_no5], 'A_cone')[0]
    p_no2 = final_pos_lean(m_kc, [sc_no2], 'A_cone')[0]
    assert (p_full[0] - p_no5[0]).norm().item() < 1e-6, 'A_cone must have exact 0 shift outside cone'
    assert (p_full[0] - p_no2[0]).norm().item() > 1e-5, 'A_cone must respond to >3m head-on approaching ped 2'

    # 2. Verify TF-Rad1 vs TF-Rad2 vs TF-Cone1 exact Theorem 1 horizon laws!
    m_r1 = make_fast_model('tf_rad1').eval()
    m_r2 = make_fast_model('tf_rad2').eval()
    m_c1 = make_fast_model('tf_cone1').eval()
    m_pe = make_fast_model('tf_pe').eval()

    # TF-Rad1: exact 0 on ped 3 (4.6m > 3m), ped 2 (3.6m > 3m), and ped 5 (25m)
    r1_full = final_pos_lean(m_r1, [sc_full], 'A_eye')[0]
    r1_no3 = final_pos_lean(m_r1, [sc_no3], 'A_eye')[0]
    r1_no2 = final_pos_lean(m_r1, [sc_no2], 'A_eye')[0]
    assert (r1_full[0] - r1_no3[0]).norm().item() < 1e-6, '1-layer 3m TF must be strictly 0 at 4.6m'
    assert (r1_full[0] - r1_no2[0]).norm().item() < 1e-6, '1-layer 3m TF misses >3m head-on ped 2'

    # TF-Rad2 (2 layers): NON-ZERO on ped 3 (4.6m via 2-hop ped 4!), EXACT 0 on ped 5 (25m > 6m)!
    r2_full = final_pos_lean(m_r2, [sc_full], 'A_eye')[0]
    r2_no3 = final_pos_lean(m_r2, [sc_no3], 'A_eye')[0]
    r2_no5 = final_pos_lean(m_r2, [sc_no5], 'A_eye')[0]
    assert (r2_full[0] - r2_no3[0]).norm().item() > 1e-5, '2-layer 3m TF must exhibit 2-hop sensitivity at 4.6m'
    assert (r2_full[0] - r2_no5[0]).norm().item() < 1e-6, '2-layer 3m TF must be strictly 0 beyond 6m'

    # TF-Cone1: NON-ZERO on >3m head-on ped 2, EXACT 0 on non-approaching ped 3 (4.6m) & ped 5 (25m), AND 0 order shift!
    c1_full = final_pos_lean(m_c1, [sc_full], 'A_eye')[0]
    c1_no2 = final_pos_lean(m_c1, [sc_no2], 'A_eye')[0]
    c1_no3 = final_pos_lean(m_c1, [sc_no3], 'A_eye')[0]
    c1_no5 = final_pos_lean(m_c1, [sc_no5], 'A_eye')[0]
    assert (c1_full[0] - c1_no2[0]).norm().item() > 1e-5, 'TF-Cone1 must respond to >3m head-on ped 2'
    assert (c1_full[0] - c1_no3[0]).norm().item() < 1e-6, 'TF-Cone1 must ignore non-approaching 4.6m ped 3'
    assert (c1_full[0] - c1_no5[0]).norm().item() < 1e-6, 'TF-Cone1 must ignore 25m ped 5'
    assert order_shift(m_c1, [sc_full], 'A_eye') < 1e-6, 'TF-Cone1 must be 100% permutation equivariant'
    assert order_shift(m_pe, [sc_full], 'A_eye') > 1e-4, 'TF-IndexPE must break permutation equivariance'
    print('IJCAI PROBE SELFTEST OK: A_cone, TF-Rad1 (<=3m), TF-Rad2 (<=6m), TF-Cone1, TF-PE all mathematically verified!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--runs', help='directory of trained run folders')
    ap.add_argument('--datasets', nargs='+', default=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd'])
    ap.add_argument('--out', help='output prefix for .json and .md summary')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run_all(a.runs, a.datasets, a.out)
