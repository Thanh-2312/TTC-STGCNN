"""Universal IJCAI 2027 Colab Orchestrator:
1. Evaluates TTC-OSC Sample Coupling + JADE Dichotomy on ALL 20 pre-trained models x 5 ETH/UCY splits ('raw_id').
2. Evaluates Universal Input Wrapper ('rad3_id' vs 'cone_id') + Combined (TTC-Cone + TTC-OSC) across 11 representative models.
3. Empirically verifies exact 0.000000 m Track-ID Permutation ('cone_r1') and Outside-Cone Removal ('cone_c0') invariance.
"""
import os, sys, json, time, shutil, zipfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
BENCH_A = os.path.normpath(os.path.join(HERE, '..', 'bench_a'))
IDEA1 = os.path.normpath(os.path.join(HERE, '..', 'idea1'))
sys.path[:0] = [BENCH_A, IDEA1, HERE]

WORK = os.environ.get('UNIV_WORK', '/content/univ_work')
os.environ['BENCH_WORK'] = WORK
REPOS = os.environ.setdefault('BENCH_REPOS', '/content/repos')
PERT = os.environ.get('UNIV_PERT', '/content/univ_pert')
WORKERS = int(os.environ.get('UNIV_WORKERS', max(4, min(16, (os.cpu_count() or 8) - 2))))
DATASETS = [d for d in os.environ.get('UNIV_DS', 'eth,hotel,univ,zara1,zara2').split(',') if d]
SYM_DS = [d for d in DATASETS if d != 'univ'] or DATASETS  # fast full-coverage symmetry splits (incl. >10m support)
SMOKE = os.environ.get('UNIV_SMOKE', '') == '1'
OUT = os.path.join(WORK, 'out')
CKPT_W1 = os.path.join(WORK, 'ckpts_width1')
T0 = time.time()

import run as A  # noqa: E402
from adapters import MODELS  # noqa: E402
from run_loc import color_file_tracks, drop_tracks_in_file  # noqa: E402

WRAP_SET = {'Singular', 'ET-PECNet', 'ET-LBEBM', 'EqMotion', 'STGCNN-F', 'DMRGCN-F',
            'GraphTERN', 'Implicit', 'STGCNN', 'PECNet', 'SGCN'}


def log(*msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *msg, flush=True)


def data_root(repo, cond):
    rel = A.PINNED[repo][2]
    return os.path.join(REPOS, repo, rel) if cond == 'id' else os.path.join(PERT, repo, cond, rel)


def build_pert_sets(repos, dss):
    """Build r1 (random track-ID bijection) and c0 (colored single-pedestrian removal) for symmetry verification."""
    ref_repo = 'Social-STGCNN' if 'Social-STGCNN' in repos else repos[0]
    ref_base = data_root(ref_repo, 'id')
    c0_map = {}
    for ds in dss:
        c0_map[ds] = {os.path.basename(f).replace('.txt', ''): color_file_tracks(f, 1)
                      for f in A.test_files(ref_repo, ref_base, ds)}
    for repo in repos:
        # r1: permuted track IDs
        dst_r1 = data_root(repo, 'r1')
        if not os.path.exists(dst_r1 + '.done'):
            shutil.rmtree(dst_r1, ignore_errors=True)
            shutil.copytree(data_root(repo, 'id'), dst_r1)
            for ds in dss:
                for f in A.test_files(repo, dst_r1, ds):
                    A.remap_file(f, 1)
            open(dst_r1 + '.done', 'w').close()
        # c0: colored single removal
        dst_c0 = data_root(repo, 'c0')
        if not os.path.exists(dst_c0 + '.done'):
            shutil.rmtree(dst_c0, ignore_errors=True)
            shutil.copytree(data_root(repo, 'id'), dst_c0)
            for ds in dss:
                for f in A.test_files(repo, dst_c0, ds):
                    stem = os.path.basename(f).replace('.txt', '')
                    cs = c0_map[ds].get(stem, [])
                    if cs:
                        drop_tracks_in_file(f, cs[0])
            open(dst_c0 + '.done', 'w').close()
    log('perturbed sets (r1, c0) ready for:', dss)


def model_specs():
    specs = [(m, m, '', '') for m in MODELS]
    zpath = os.path.join(ROOT, 'data', 'idea1_locality', 'ckpts_width1.zip')
    if os.path.exists(zpath):
        if not os.path.isdir(CKPT_W1):
            os.makedirs(CKPT_W1, exist_ok=True)
            zipfile.ZipFile(zpath).extractall(CKPT_W1)
        for base in ('STGCNN', 'DMRGCN'):
            specs.append((f'{base}-F', base, 'F', CKPT_W1))
    only = [x.strip() for x in os.environ.get('UNIV_ONLY', '').split(',') if x.strip()]
    return [s for s in specs if not only or s[0] in only]


def build_job_list(specs):
    jobs = []
    ordered_ds = sorted(DATASETS, key=lambda d: d == 'univ')  # fast splits first, univ last
    # Wave 1: raw_id on ALL models (fast; gives full 20-model TTC-OSC + baseline)
    for ds in ordered_ds:
        for mname, base, var, ckroot in specs:
            ck = os.path.join(ckroot, f'{base}-F-s0_{ds}') if ckroot else ''
            jobs.append((mname, base, var, ck, ds, 'raw', 'id'))
    # Wave 2: rad3_id & cone_id on WRAP_SET models across all datasets
    wrap_specs = [s for s in specs if s[0] in WRAP_SET]
    for ds in ordered_ds:
        for mname, base, var, ckroot in wrap_specs:
            ck = os.path.join(ckroot, f'{base}-F-s0_{ds}') if ckroot else ''
            for mode in ('rad3', 'cone'):
                jobs.append((mname, base, var, ck, ds, mode, 'id'))
            if SMOKE and mname in ('Singular', 'ET-PECNet', 'EqMotion'):
                jobs.append((mname, base, var, ck, ds, 'inf', 'id'))  # equivariant R=inf sanity check vs raw
    # Wave 3: empirical symmetry verification (cone_r1 & cone_c0)
    sym_dss = DATASETS if SMOKE else SYM_DS
    for ds in sym_dss:
        for mname, base, var, ckroot in wrap_specs:
            ck = os.path.join(ckroot, f'{base}-F-s0_{ds}') if ckroot else ''
            for cond in ('r1', 'c0'):
                jobs.append((mname, base, var, ck, ds, 'cone', cond))
    return jobs


def main():
    os.makedirs(WORK, exist_ok=True)
    specs = model_specs()
    jobs = build_job_list(specs)
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, j[0], j[4], f'{j[5]}_{j[6]}.json'))]
    if not todo:
        log('all jobs already saved in WORK -> running analyzer directly!')
        subprocess.run([sys.executable, os.path.join(HERE, 'analyze_univ.py'), OUT, os.path.join(WORK, 'results')], check=False)
        return

    A.setup()
    wrap_repos = sorted({MODELS[base][0] for m, base, _, _ in specs if m in WRAP_SET} | {'Social-STGCNN'})
    build_pert_sets(wrap_repos, DATASETS if SMOKE else SYM_DS)

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
            mname, base, var, ck, ds, mode, cond = todo.pop(0)
            key = f'{mname}_{ds}_{mode}_{cond}'
            cmd = [sys.executable, os.path.join(HERE, 'job_univ.py'),
                   '--name', mname, '--model', base, '--ds', ds, '--mode', mode, '--cond', cond,
                   '--root', data_root(MODELS[base][0], cond), '--out', OUT]
            if var:
                cmd += ['--variant', var, '--ckpt', ck]
            lf = open(os.path.join(WORK, 'logs', f'{key}.txt'), 'w', encoding='utf-8')
            running[key] = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if time.time() - beat > 90:
            beat = time.time()
            log(f'progress: running={len(running)}, queued={len(todo)}, failed={len(failed)}')
        time.sleep(1.5)

    json.dump(failed, open(os.path.join(WORK, 'failed.json'), 'w'), indent=1)
    log('all jobs finished; failed:', len(failed))
    subprocess.run([sys.executable, os.path.join(HERE, 'analyze_univ.py'), OUT, os.path.join(WORK, 'results')], check=False)


if __name__ == '__main__':
    main()
