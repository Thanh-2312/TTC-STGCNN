"""Idea 1 orchestrator (Spatial Locality / Agent Removal across 20 models: 18 released + STGCNN-F & DMRGCN-F width-1).

Reuses A-bench's pinned repo setup, constructs co-occurrence graph-colored single-removal test sets (guaranteeing at
most 1 removed pedestrian in every N>=3 20-frame scene and 0 emptied frames), runs resumable parallel GPU jobs into
LOC_WORK (on Google Drive), and generates tables + figures via analyze_loc.py.

Env knobs:
  LOC_WORK     : persistent work/output folder (keep on Google Drive; re-running resumes)
  BENCH_REPOS  : local SSD folder for cloned model repos (default: /content/repos)
  LOC_PERT     : local SSD folder for colored removal test sets (default: /content/loc_pert)
  LOC_WORKERS  : parallel subprocesses on GPU (default: auto from CPU count, capped at 24)
  LOC_DS       : comma-separated datasets (default: eth,hotel,univ,zara1,zara2)
  LOC_COLORS   : max removal color sets per dataset (default: 12; covers 100% of eth & ~200k pairs overall)
  LOC_ONLY     : optional comma-separated subset of model names (for debugging/smoke)
"""
import os, sys, json, time, shutil, zipfile, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
BENCH_A = os.path.normpath(os.path.join(HERE, '..', 'bench_a'))
sys.path[:0] = [BENCH_A, HERE]

WORK = os.environ.get('LOC_WORK', '/content/idea1_work')
os.environ['BENCH_WORK'] = WORK
REPOS = os.environ.setdefault('BENCH_REPOS', '/content/repos')
PERT = os.environ.get('LOC_PERT', '/content/loc_pert')
WORKERS = int(os.environ.get('LOC_WORKERS', max(4, min(24, (os.cpu_count() or 8) - 2))))
DATASETS = [d for d in os.environ.get('LOC_DS', 'eth,hotel,univ,zara1,zara2').split(',') if d]
MAX_COLORS = int(os.environ.get('LOC_COLORS', 12))
OUT = os.path.join(WORK, 'out')
CKPT_W1 = os.path.join(WORK, 'ckpts_width1')
T0 = time.time()

import run as A  # noqa: E402  (PINNED, RELEASES, EQ_TEST, setup, test_files)
from adapters import MODELS  # noqa: E402


def log(*msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *msg, flush=True)


def color_file_tracks(path, max_colors):
    """Return list of track-ID sets (one per color, up to max_colors) such that no two tracks of the same color ever
    co-occur in any 20-frame window with N>=3 full-window pedestrians."""
    lines = [l for l in open(path, encoding='utf-8').read().splitlines() if l.strip()]
    rows = np.array([[float(x) for x in l.split()] for l in lines])
    frames = sorted(np.unique(rows[:, 0]).tolist())
    fidx = {f: i for i, f in enumerate(frames)}
    fdata = [rows[rows[:, 0] == f] for f in frames]
    cands, win_all = set(), []
    for idx in range(len(frames) - 20 + 1):
        seq = np.concatenate(fdata[idx:idx + 20], 0)
        all_p = {int(p) for p in np.unique(seq[:, 1])}
        full = []
        for p in sorted(all_p):
            c = seq[seq[:, 1] == p]
            if len(c) == 20 and fidx[c[-1, 0]] - fidx[c[0, 0]] + 1 == 20:
                full.append(p)
        if len(full) >= 3:
            cands.update(full)
            win_all.append(all_p)
    adj = {p: set() for p in cands}
    for all_p in win_all:
        present = all_p & cands
        for p in present:
            adj[p].update(present - {p})
    colors = {}
    for u in sorted(adj, key=lambda x: (-len(adj[x]), x)):
        used = {colors[v] for v in adj[u] if v in colors}
        c = 0
        while c in used:
            c += 1
        colors[u] = c
    n_col = min(max_colors, (max(colors.values()) + 1) if colors else 0)
    return [{p for p, c in colors.items() if c == k} for k in range(n_col)]


def drop_tracks_in_file(path, drop_ids):
    """Remove rows with track ID in drop_ids while preserving every frame in np.unique(frames)."""
    if not drop_ids:
        return
    lines = [l for l in open(path, encoding='utf-8').read().splitlines() if l.strip()]
    delim = '\t' if '\t' in lines[0] else ' '
    parsed = [(float((l.strip().split(delim) if delim == '\t' else l.split())[0]),
               int(float((l.strip().split(delim) if delim == '\t' else l.split())[1])), l) for l in lines]
    by_frame, order = {}, []
    for f, pid, raw in parsed:
        if f not in by_frame:
            by_frame[f] = []
            order.append(f)
        by_frame[f].append((pid, raw))
    out_lines = []
    for f in order:
        frows = by_frame[f]
        kept = [raw for pid, raw in frows if pid not in drop_ids]
        if not kept:  # solo frame outside any N>=2 window: keep 1 row so frame count never shifts
            kept = [frows[0][1]]
        out_lines.extend(kept)
    with open(path, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(out_lines) + '\n')


def loc_root(repo, cond):
    rel = A.PINNED[repo][2]
    return os.path.join(REPOS, repo, rel) if cond in ('id', 'idrep') else os.path.join(PERT, repo, cond, rel)


def build_colored_sets(repos):
    """Build deterministic colored removal copies across all requested repos & datasets; return conds per dataset."""
    ref_repo = 'Social-STGCNN' if 'Social-STGCNN' in repos else repos[0]
    ref_base = loc_root(ref_repo, 'id')
    schedule = {}
    for ds in DATASETS:
        files = A.test_files(ref_repo, ref_base, ds)
        per_file = {os.path.basename(f).replace('.txt', ''): color_file_tracks(f, MAX_COLORS) for f in files}
        n_col = max((len(v) for v in per_file.values()), default=0)
        schedule[ds] = (n_col, per_file)

    max_c = max((v[0] for v in schedule.values()), default=0)
    for repo in repos:
        for c in range(max_c):
            cond = f'c{c}'
            dst = loc_root(repo, cond)
            if os.path.exists(dst + '.done'):
                continue
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(loc_root(repo, 'id'), dst)
            for ds in DATASETS:
                _, per_file = schedule[ds]
                for f in A.test_files(repo, dst, ds):
                    stem = os.path.basename(f).replace('.txt', '')
                    csets = per_file.get(stem, [])
                    if c < len(csets):
                        drop_tracks_in_file(f, csets[c])
            open(dst + '.done', 'w').close()
    conds_by_ds = {ds: ['id', 'idrep'] + [f'c{c}' for c in range(schedule[ds][0])] for ds in DATASETS}
    json.dump({ds: dict(n_colors=schedule[ds][0], conds=conds_by_ds[ds]) for ds in DATASETS},
              open(os.path.join(WORK, 'schedule.json'), 'w'), indent=1)
    log('colored removal test sets ready:', {ds: schedule[ds][0] for ds in DATASETS})
    return conds_by_ds


def model_specs():
    """All 18 released models + 2 width-1 retrained models (STGCNN-F, DMRGCN-F) if ckpts_width1.zip is present."""
    specs = [(m, m, '', '') for m in MODELS]
    zpath = os.path.join(ROOT, 'data', 'idea1_locality', 'ckpts_width1.zip')
    if os.path.exists(zpath):
        if not os.path.isdir(CKPT_W1):
            os.makedirs(CKPT_W1, exist_ok=True)
            zipfile.ZipFile(zpath).extractall(CKPT_W1)
        for base in ('STGCNN', 'DMRGCN'):
            specs.append((f'{base}-F', base, 'F', CKPT_W1))
    only = [x.strip() for x in os.environ.get('LOC_ONLY', '').split(',') if x.strip()]
    return [s for s in specs if not only or s[0] in only]


def main():
    os.makedirs(WORK, exist_ok=True)
    specs = model_specs()
    A.setup()
    repos = sorted({MODELS[base][0] for _, base, _, _ in specs} | {'Social-STGCNN'})
    conds_by_ds = build_colored_sets(repos)

    jobs = []
    for ds in sorted(DATASETS, key=lambda d: d == 'univ'):  # fast datasets first, univ last
        for mname, base, var, ckroot in specs:
            ck = os.path.join(ckroot, f'{base}-F-s0_{ds}') if ckroot else ''
            for c in conds_by_ds[ds]:
                jobs.append((mname, base, var, ck, ds, c))

    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, j[0], j[4], j[5] + '.json'))]
    log(f'total jobs: {len(jobs)}, remaining: {len(todo)}, parallel workers: {WORKERS}')

    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    running, failed, beat = {}, [], time.time()
    os.makedirs(os.path.join(WORK, 'logs'), exist_ok=True)

    while todo or running:
        for key, p in list(running.items()):
            if p.poll() is not None:
                del running[key]
                if p.returncode:
                    failed.append(key)
                    log('FAILED', key, 'rc', p.returncode)
        while todo and len(running) < WORKERS:
            mname, base, var, ck, ds, c = todo.pop(0)
            key = f'{mname}_{ds}_{c}'
            cmd = [sys.executable, os.path.join(HERE, 'job_loc.py'),
                   '--name', mname, '--model', base, '--ds', ds, '--cond', c,
                   '--root', loc_root(MODELS[base][0], c), '--out', OUT]
            if var:
                cmd += ['--variant', var, '--ckpt', ck]
            lf = open(os.path.join(WORK, 'logs', f'{key}.txt'), 'w', encoding='utf-8')
            running[key] = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if time.time() - beat > 120:
            beat = time.time()
            log(f'progress: running={len(running)}, queued={len(todo)}, failed={len(failed)}')
        time.sleep(1.5)

    json.dump(failed, open(os.path.join(WORK, 'failed.json'), 'w'), indent=1)
    log('all jobs finished; failed:', len(failed))
    ref_json = os.path.join(ROOT, 'data', 'idea1_locality', 'fast_stgcnn_ref.json')
    subprocess.run([sys.executable, os.path.join(HERE, 'analyze_loc.py'), OUT, os.path.join(WORK, 'results'),
                    '--ref', ref_json], check=False)


if __name__ == '__main__':
    main()
