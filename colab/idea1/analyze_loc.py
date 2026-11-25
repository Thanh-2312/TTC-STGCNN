"""Analyze Idea-1 Spatial Locality results across all evaluated models, generate Markdown/JSON tables & PDF/PNG plots.

Usage:
  python colab/idea1/analyze_loc.py <out_dir> <results_prefix> [--ref data/idea1_locality/fast_stgcnn_ref.json]
  python colab/idea1/analyze_loc.py --selftest   # local invariant & signature-matching self-check on ETH/UCY raw files
"""
import argparse, os, sys, glob, json, hashlib, zipfile, tempfile
import numpy as np

DS_ALL = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
D_BINS = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
V_BINS = [0.0, 0.05, 0.1, 0.2, 0.4, np.inf]

FAMILY = {
    'Singular': ('Diffusion (CVPR 24)', 'I'),
    'EqMotion': ('Equiv-GNN (CVPR 23)', 'I'),
    'ET-AgentFormer': ('Transformer (ICCV 23)', 'I'),
    'PECNet': ('Goal-Pool (ECCV 20)', 'I'),
    'ET-PECNet': ('Goal-Eigen (ICCV 23)', 'I'),
    'ET-LBEBM': ('Energy-Eigen (ICCV 23)', 'I'),
    'STGCNN-F': ('GCN 3x1 (Order-Inv)', 'F'),
    'DMRGCN-F': ('DMRGCN 3x1 (Order-Inv)', 'F'),
    'STGCNN': ('GCN 3x3 (CVPR 20)', 'D'),
    'DMRGCN': ('DMRGCN 3x3 (AAAI 21)', 'D'),
    'GraphTERN': ('Graph-TERN (AAAI 23)', 'D'),
    'Implicit': ('Social-Implicit (ECCV 22)', 'D'),
    'SGCN': ('Sparse-GCN (CVPR 21)', 'D'),
    'ET-STGCNN': ('ET-STGCNN (ICCV 23)', 'D'),
    'ET-SGCN': ('ET-SGCN (ICCV 23)', 'D'),
    'ET-DMRGCN': ('ET-DMRGCN (ICCV 23)', 'D'),
    'ET-GPGraph-STGCNN': ('ET-GPGraph-STG (ICCV 23)', 'D'),
    'ET-GPGraph-SGCN': ('ET-GPGraph-SGCN (ICCV 23)', 'D'),
    'ET-GraphTERN': ('ET-GraphTERN (ICCV 23)', 'D'),
    'ET-Implicit': ('ET-Implicit (ICCV 23)', 'D'),
}


def scene_sig(keys_1d):
    return hashlib.blake2b(np.sort(keys_1d).tobytes(), digest_size=16).digest()


def unpack_scenes(npz_path):
    z = np.load(npz_path)
    sizes, keys, pins = z['sizes'], z['key'], z['pin']
    obs = z['obs'] if 'obs' in z.files else None
    offs = np.cumsum([0, *sizes.tolist()])
    out = []
    for i in range(len(sizes)):
        a, b = offs[i], offs[i + 1]
        out.append((keys[a:b], pins[a:b], obs[a:b] if obs is not None else None))
    return out, z


def binned(vals, key, bins):
    idx = np.digitize(key, bins) - 1
    return [dict(lo=float(bins[b]), hi=float(bins[b + 1]), n=int((idx == b).sum()),
                 mean=float(vals[idx == b].mean()) if (idx == b).any() else None)
            for b in range(len(bins) - 1)]


def analyze_cell(cell_dir):
    id_p = os.path.join(cell_dir, 'id.npz')
    cs = sorted(glob.glob(os.path.join(cell_dir, 'c*.npz')), key=lambda p: int(os.path.basename(p)[1:-4]))
    if not os.path.exists(id_p) or not cs:
        return None
    base_scenes, z_id = unpack_scenes(id_p)
    id_meta = json.load(open(os.path.join(cell_dir, 'id.json')))

    floor = None
    rep_p = os.path.join(cell_dir, 'idrep.npz')
    if os.path.exists(rep_p):
        z_rep = np.load(rep_p)
        if len(z_rep['pin']) == len(z_id['pin']) and np.array_equal(z_rep['key'], z_id['key']):
            floor = float(np.linalg.norm(z_rep['pin'][:, -1] - z_id['pin'][:, -1], axis=-1).mean())

    full_sigs = {scene_sig(ks) for ks, _, _ in base_scenes}
    cand_map, dup_sigs = {}, set()
    for w, (ks, pin, obs) in enumerate(base_scenes):
        if len(ks) < 3 or len(set(ks.tolist())) < len(ks):
            continue
        for j in range(len(ks)):
            rem = np.delete(ks, j)
            s = scene_sig(rem)
            if s in full_sigs or s in dup_sigs:
                continue
            if s in cand_map:
                del cand_map[s]
                dup_sigs.add(s)
            else:
                cand_map[s] = (w, j)

    ch_end, ch_ade, dd_min, dd_last, vv, ranks = [], [], [], [], [], []
    matched_sigs = set()

    for cp in cs:
        sub_scenes, _ = unpack_scenes(cp)
        for ks_sub, pin_sub, _ in sub_scenes:
            s = scene_sig(ks_sub)
            if s not in cand_map or s in matched_sigs:
                continue
            matched_sigs.add(s)
            w, j = cand_map[s]
            ks_b, pin_b, obs_b = base_scenes[w]
            keep = [i for i in range(len(ks_b)) if i != j]
            ord_b = np.argsort(ks_b[keep])
            ord_s = np.argsort(ks_sub)
            if not np.array_equal(ks_b[keep][ord_b], ks_sub[ord_s]):
                continue
            pb = pin_b[keep][ord_b]       # (N-1, 12, 2)
            ps = pin_sub[ord_s]           # (N-1, 12, 2)
            ob_rem = obs_b[keep][ord_b]   # (N-1, 8, 2)
            ob_j = obs_b[j]               # (8, 2)

            # Closest observed distance & last-frame distance
            d_t = np.linalg.norm(ob_rem - ob_j[None], axis=-1)   # (N-1, 8)
            d_min = d_t.min(axis=-1)
            d_last = d_t[:, -1]

            # Mean velocity difference per step during observation
            v_all = obs_b[:, 1:] - obs_b[:, :-1]                 # (N, 7, 2)
            v_rem = v_all[keep][ord_b]
            dv = np.linalg.norm(v_rem - v_all[j][None], axis=-1).mean(axis=-1)

            # Neighbor distance rank of removed j for each remaining i at last observed step (vectorized)
            all_last = obs_b[:, -1]                              # (N, 2)
            idx_orig = np.asarray(keep, dtype=np.intp)[ord_b]    # (N-1,)
            d_mat = np.linalg.norm(all_last[idx_orig, None] - all_last[None], axis=-1)
            d_mat[np.arange(len(idx_orig)), idx_orig] = -1.0
            rk = (d_mat < d_last[:, None]).sum(axis=-1).astype(np.int32)

            ch_end.append(np.linalg.norm(ps[:, -1] - pb[:, -1], axis=-1))
            ch_ade.append(np.linalg.norm(ps - pb, axis=-1).mean(axis=-1))
            dd_min.append(d_min); dd_last.append(d_last); vv.append(dv); ranks.append(rk)

    if not ch_end:
        return None
    ch = np.concatenate(ch_end)
    ca = np.concatenate(ch_ade)
    dm = np.concatenate(dd_min)
    dl = np.concatenate(dd_last)
    dv = np.concatenate(vv)
    rk = np.concatenate(ranks)
    far_m = dm > 5.0

    return dict(
        n_subscenes=len(matched_sigs),
        n_pairs=int(len(ch)),
        floor=floor,
        ADE=id_meta.get('ade_all'),
        FDE=id_meta.get('fde_all'),
        mean=float(ch.mean()),
        mean_ade_shift=float(ca.mean()),
        near=float(ch[dm < 2.0].mean()) if (dm < 2.0).any() else None,
        mid_3_5=float(ch[(dm >= 3.0) & (dm <= 5.0)].mean()) if ((dm >= 3.0) & (dm <= 5.0)).any() else None,
        far=float(ch[far_m].mean()) if far_m.any() else None,
        vfar_10=float(ch[dm > 10.0].mean()) if (dm > 10.0).any() else None,
        max_gt3=float(ch[dm > 3.0].max()) if (dm > 3.0).any() else 0.0,
        share_gt_10cm=float((ch > 0.1).mean()),
        share_far_gt_5cm=float((ch[far_m] > 0.05).mean()) if far_m.any() else 0.0,
        by_dist=binned(ch, dm, D_BINS),
        by_dv_all=binned(ch, dv, V_BINS),
        by_dv_far=binned(ch[far_m], dv[far_m], V_BINS) if far_m.any() else [],
        by_rank_last={str(r): (float(ch[rk == r].mean()) if (rk == r).any() else None) for r in (1, 2, 3)}
                     | {'4+': float(ch[rk >= 4].mean()) if (rk >= 4).any() else None},
    )


def eq_weight(cells_for_model, key):
    vals = [v[key] for ds, v in cells_for_model.items() if v.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def eq_curve(cells_for_model, field='by_dist', exclude_eth_10=False):
    bins_out = []
    sample_bins = next((v[field] for v in cells_for_model.values() if v.get(field)), [])
    for b_idx, b_meta in enumerate(sample_bins):
        dss = [d for d in DS_ALL if not (exclude_eth_10 and d == 'eth')]
        vals = [cells_for_model[d][field][b_idx]['mean']
                for d in dss if d in cells_for_model and b_idx < len(cells_for_model[d][field])
                and cells_for_model[d][field][b_idx]['mean'] is not None]
        bins_out.append((b_meta['lo'], b_meta['hi'], float(np.mean(vals)) if vals else None))
    return bins_out


def plot_overview(summary, res_prefix):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib skipped:', e)
        return

    focus = [
        ('Singular', 'SingularTrajectory (Diffusion, CVPR 24)', '#8e44ad', 'o', '-'),
        ('ET-AgentFormer', 'ET-AgentFormer (Transformer, ICCV 23)', '#d35400', '^', '-'),
        ('EqMotion', 'EqMotion (Equiv-GNN, CVPR 23)', '#16a085', 's', '-'),
        ('PECNet', 'PECNet (Social Pool, ECCV 20)', '#2980b9', 'D', '-'),
        ('STGCNN-F', 'Social-STGCNN 3x1 (Order-Inv GCN)', '#2c3e50', 'v', '--'),
        ('DMRGCN-F', 'DMRGCN 3x1 (Order-Inv GCN)', '#7f8c8d', 'P', '--'),
        ('FastSTGCNN-KN', 'Ours: 3 m Radius + Receiver Norm (KN)', '#b7950b', '*', ':'),
    ]
    present = [f for f in focus if f[0] in summary and summary[f[0]].get('curve_4ds')]
    if not present:
        return

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    # Panel A: consistent 4-split curve (hotel, univ, zara1, zara2 — all bins populated up to >10 m)
    for m, lab, col, mk, ls in present:
        pts = summary[m]['curve_4ds']
        xs = [i for i, (_, _, y) in enumerate(pts) if y is not None]
        ys = [y for _, _, y in pts if y is not None]
        ax[0].plot(xs, ys, ls=ls, color=col, marker=mk, lw=1.8, markersize=5, label=lab)
    lbls = [f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'>{lo:g}' for lo, hi, _ in summary[present[0][0]]['curve_4ds']]
    ax[0].set_xticks(range(len(lbls))); ax[0].set_xticklabels(lbls, fontsize=8)
    ax[0].set_xlabel('Closest observed distance to removed pedestrian (m)', fontsize=8.5)
    ax[0].set_ylabel('Shift of final predicted position (m)', fontsize=8.5)
    ax[0].set_title('Spatial Sensitivity Profile (4 splits with >10 m range)', fontsize=9)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=6.8, loc='upper right')

    # Panel B: Near (<2 m) vs Far (>5 m) bar chart across all evaluated Order-Invariant + Reference models
    bar_models = [m for m in summary if summary[m]['cls'] in ('I', 'F', 'Ref') and summary[m]['near_eq'] is not None]
    x = np.arange(len(bar_models))
    w = 0.38
    ax[1].bar(x - w / 2, [summary[m]['near_eq'] for m in bar_models], w, label='Near (< 2 m)', color='#2471a3')
    ax[1].bar(x + w / 2, [summary[m]['far_eq'] or 0.0 for m in bar_models], w, label='Far (> 5 m)', color='#c0392b')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(bar_models, rotation=35, ha='right', fontsize=7.5)
    ax[1].set_ylabel('Mean endpoint shift (m, 5-split equal weight)', fontsize=8.5)
    ax[1].set_title('Near vs. Far Sensitivity in Order-Invariant Predictors', fontsize=9)
    ax[1].grid(alpha=0.3, axis='y')
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{res_prefix}_overview.{ext}', dpi=220)
    plt.close(fig)


def run_analysis(out_dir, res_prefix, ref_path=None):
    cells = {}
    for d in sorted(glob.glob(os.path.join(out_dir, '*', '*'))):
        if not os.path.isdir(d):
            continue
        m, ds = os.path.basename(os.path.dirname(d)), os.path.basename(d)
        c = analyze_cell(d)
        if c:
            cells.setdefault(m, {})[ds] = c
            print(f"  [analyze] {m:18s} {ds:6s} -> {c['n_pairs']:6d} pairs", flush=True)

    if ref_path and os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        for m in ('FastSTGCNN-K', 'FastSTGCNN-K0', 'FastSTGCNN-KP', 'FastSTGCNN-KR', 'FastSTGCNN-KN'):
            if m in ref:
                cells[m] = ref[m]

    order = [m for m in list(FAMILY) + ['FastSTGCNN-K', 'FastSTGCNN-K0', 'FastSTGCNN-KP', 'FastSTGCNN-KR', 'FastSTGCNN-KN']
             if m in cells] + [m for m in sorted(cells) if m not in FAMILY and not m.startswith('FastSTGCNN-')]

    summary = {}
    for m in order:
        fam, cls = FAMILY.get(m, ('Controlled FastSTGCNN', 'Ref'))
        near_eq = eq_weight(cells[m], 'near')
        far_eq = eq_weight(cells[m], 'far')
        summary[m] = dict(
            family=fam, cls=cls, datasets=sorted(cells[m].keys()),
            ADE_eq=eq_weight(cells[m], 'ADE'),
            near_eq=near_eq,
            far_eq=far_eq,
            far_near_ratio=(far_eq / near_eq) if (near_eq and far_eq is not None and near_eq > 1e-7) else 0.0,
            max_gt3=max((v.get('max_gt3', 0.0) for v in cells[m].values()), default=0.0),
            share10_eq=eq_weight(cells[m], 'share_gt_10cm'),
            floor_max=max((v['floor'] for v in cells[m].values() if v.get('floor') is not None), default=None),
            curve_5ds=eq_curve(cells[m], 'by_dist', exclude_eth_10=False),
            curve_4ds=eq_curve(cells[m], 'by_dist', exclude_eth_10=True),
        )

    os.makedirs(os.path.dirname(os.path.abspath(res_prefix)), exist_ok=True)
    json.dump(dict(summary=summary, cells=cells), open(res_prefix + '.json', 'w'), indent=1)

    L = [
        '# Idea 1 — Spatial Locality Benchmark Across Trajectory Prediction Families',
        '',
        'Equal-weight mean across ETH/UCY splits (`Near`: closest distance `< 2 m`; `Far`: `> 5 m`; `Ratio`: `Far / Near`; `>10cm`: % of remaining agents shifting by `> 0.1 m`).',
        '',
        '| Model | Architecture Family | Group | Splits | minADE20 | Near <2m (m) | Far >5m (m) | Far/Near | Max >3m (m) | >0.1m (%) | Repeat Floor (m) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    fmt = lambda v, f='{:.4f}': f.format(v) if v is not None else '—'
    for m in order:
        s = summary[m]
        L.append(f"| {m} | {s['family']} | {s['cls']} | {len(s['datasets'])} | {fmt(s['ADE_eq'], '{:.3f}')} | "
                 f"{fmt(s['near_eq'])} | {fmt(s['far_eq'])} | {fmt(s['far_near_ratio'], '{:.2f}')} | "
                 f"{fmt(s['max_gt3'], '{:.2e}')} | {fmt(100 * s['share10_eq'] if s['share10_eq'] is not None else None, '{:.1f}%')} | "
                 f"{fmt(s['floor_max'], '{:.1e}')} |")

    L += ['', '## Distance Bins (4 splits with full >10 m support: hotel, univ, zara1, zara2)', '',
          '| Model | 0-1 m | 1-2 m | 2-3 m | 3-5 m | 5-10 m | >10 m |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in order:
        pts = summary[m]['curve_4ds'] if any(y is not None for _, _, y in summary[m]['curve_4ds']) else summary[m]['curve_5ds']
        if pts:
            L.append(f"| {m} | " + ' | '.join(fmt(y) for _, _, y in pts) + ' |')

    open(res_prefix + '.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    plot_overview(summary, res_prefix)
    print('\n'.join(L[:min(len(L), 32)]))


def selftest():
    """Runnable invariant check (ponytail self-check): coloring, frame preservation, 0 multi-drop, 100% matching."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, '..', '..'))
    sys.path.insert(0, here)
    from run_loc import color_file_tracks, drop_tracks_in_file
    from job_loc import pkeys

    zraw = os.path.join(root, 'data', 'idea1_locality', 'eth_ucy_test_raw.zip')
    assert os.path.exists(zraw), f'Missing {zraw}'
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zraw).extractall(tmp)
        for ds in ('eth', 'hotel', 'zara1'):
            f = sorted(glob.glob(os.path.join(tmp, ds, '*.txt')))[0]
            orig_lines = [l for l in open(f).read().splitlines() if l.strip()]
            orig_rows = np.array([[float(x) for x in l.split()] for l in orig_lines])
            orig_frames = sorted(np.unique(orig_rows[:, 0]).tolist())
            csets = color_file_tracks(f, 4)
            assert len(csets) >= 2, ds

            # Verify: in every N>=3 window, at most 1 track belongs to any color set, and frames never vanish
            fidx = {fr: i for i, fr in enumerate(orig_frames)}
            fdata = [orig_rows[orig_rows[:, 0] == fr] for fr in orig_frames]
            for cset in csets:
                for idx in range(len(orig_frames) - 19):
                    seq = np.concatenate(fdata[idx:idx + 20], 0)
                    full = [int(p) for p in np.unique(seq[:, 1])
                            if len(seq[seq[:, 1] == p]) == 20 and fidx[seq[seq[:, 1] == p][-1, 0]] - fidx[seq[seq[:, 1] == p][0, 0]] + 1 == 20]
                    if len(full) >= 3:
                        assert len(set(full) & cset) <= 1, (ds, len(set(full) & cset))

                tf = os.path.join(tmp, f'{ds}_pert.txt')
                open(tf, 'w').write('\n'.join(orig_lines) + '\n')
                drop_tracks_in_file(tf, cset)
                prows = np.array([[float(x) for x in l.split()] for l in open(tf).read().splitlines() if l.strip()])
                assert sorted(np.unique(prows[:, 0]).tolist()) == orig_frames, f'Frame dropped in {ds}!'
                # Verify upstream TrajectoryDataset invariant: no gapped track with pad_end - pad_front == 20 and len < 20
                pfd = [prows[prows[:, 0] == fr] for fr in orig_frames]
                for idx in range(len(orig_frames) - 19):
                    seq = np.concatenate(pfd[idx:idx + 20], 0)
                    for pid in np.unique(seq[:, 1]):
                        c_ = seq[seq[:, 1] == pid]
                        if fidx[c_[-1, 0]] - fidx[c_[0, 0]] + 1 == 20:
                            assert len(c_) == 20, f'Gapped track {pid} in {ds}: span=20 but rows={len(c_)}!'

        # Verify synthetic cell analysis end-to-end (synthetic non-local model vs exact 3m local model)
        out_mock = os.path.join(tmp, 'out')
        for mname, nonlocal_w in [('MockGlobal', 0.2), ('MockLocal3m', 0.0)]:
            mdir = os.path.join(out_mock, mname, 'eth')
            os.makedirs(mdir, exist_ok=True)
            f = os.path.join(tmp, 'eth', 'biwi_eth.txt')
            csets = color_file_tracks(f, 5)

            def eval_txt(txt_path, cond):
                rows = np.array([[float(x) for x in l.split()] for l in open(txt_path).read().splitlines() if l.strip()])
                frs = sorted(np.unique(rows[:, 0]).tolist())
                fi = {fr: i for i, fr in enumerate(frs)}
                fd = [rows[rows[:, 0] == fr] for fr in frs]
                sz, ka, pa, oa = [], [], [], []
                for idx in range(len(frs) - 19):
                    seq = np.concatenate(fd[idx:idx + 20], 0)
                    full = [seq[seq[:, 1] == p][:, 2:] for p in sorted(np.unique(seq[:, 1]))
                            if len(seq[seq[:, 1] == p]) == 20 and fi[seq[seq[:, 1] == p][-1, 0]] - fi[seq[seq[:, 1] == p][0, 0]] + 1 == 20]
                    if len(full) < 2:
                        continue
                    tr = np.stack(full, 0).astype(np.float32)  # N, 20, 2
                    obs, gt = tr[:, :8], tr[:, 8:]
                    # Mock prediction: last obs + interaction from neighbors (<3m strong, >3m controlled by nonlocal_w)
                    n = len(obs)
                    dmat = np.linalg.norm(obs[:, None] - obs[None], axis=-1).min(-1)
                    wmat = np.where(dmat < 3.0, 0.3, nonlocal_w)
                    np.fill_diagonal(wmat, 0.0)
                    shift_vec = (wmat[:, :, None] * obs[:, None, -1]).sum(1) / max(1, 1)  # unnormalized sum changes on removal
                    if nonlocal_w == 0.0:  # receiver-degree normalized 3m kernel -> strictly 0 change beyond 3m!
                        deg = (dmat < 3.0).sum(1, keepdims=True)
                        w_loc = np.where((dmat < 3.0) & (dmat > 0), 0.3, 0.0)
                        shift_vec = (w_loc[:, :, None] * obs[:, None, -1]).sum(1) / deg
                    pin = np.repeat((obs[:, -1:] + shift_vec[:, None]), 12, axis=1).astype(np.float32)
                    sz.append(n); ka.append(pkeys(obs, gt)); pa.append(pin); oa.append(obs)
                pay = dict(sizes=np.array(sz, np.int32), key=np.concatenate(ka), pin=np.concatenate(pa))
                if cond == 'id':
                    pay.update(obs=np.concatenate(oa), ade=np.zeros(sum(sz), np.float32), fde=np.zeros(sum(sz), np.float32))
                np.savez_compressed(os.path.join(mdir, f'{cond}.npz'), **pay)
                json.dump(dict(ade_all=0.5, fde_all=1.0), open(os.path.join(mdir, f'{cond}.json'), 'w'))

            eval_txt(f, 'id')
            eval_txt(f, 'idrep')
            for ci, cs_ in enumerate(csets):
                tf = os.path.join(tmp, f'eth_c{ci}.txt')
                open(tf, 'w').write(open(f).read())
                drop_tracks_in_file(tf, cs_)
                eval_txt(tf, f'c{ci}')

        run_analysis(out_mock, os.path.join(tmp, 'res'))
        res = json.load(open(os.path.join(tmp, 'res.json')))['summary']
        assert res['MockGlobal']['far_eq'] > 0.05, res['MockGlobal']
        assert res['MockLocal3m']['max_gt3'] < 1e-6, res['MockLocal3m']
        assert res['MockLocal3m']['floor_max'] == 0.0
    print('SELFTEST PASSED: 0 empty frames, <=1 drop per N>=3 scene, MockGlobal far>0.05m, MockLocal3m max_gt3<1e-6m!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir', nargs='?')
    ap.add_argument('res_prefix', nargs='?')
    ap.add_argument('--ref', default='')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        assert args.out_dir and args.res_prefix, 'Usage: analyze_loc.py <out_dir> <res_prefix> [--ref ...]'
        run_analysis(args.out_dir, args.res_prefix, args.ref or None)
