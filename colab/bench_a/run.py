"""A-bench orchestrator (experiments/A-bench/protocol.md): clone the pinned repositories, fetch the released checkpoints,
write the re-indexed test sets, run every (model, dataset, condition) job in parallel processes on one GPU, then analyse.

Env: BENCH_WORK (results, keep it on Drive: re-running resumes), BENCH_REPOS (clones), BENCH_PERT (re-indexed copies),
BENCH_WORKERS (parallel jobs), BENCH_ONLY (comma list of models, for a smoke run), BENCH_DS, BENCH_DRAWS.
"""
import os, sys, json, time, shutil, subprocess, zlib, urllib.request, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('BENCH_WORK', '/content/bench_a_work')
REPOS = os.environ.setdefault('BENCH_REPOS', '/content/repos')
PERT = os.environ.get('BENCH_PERT', '/content/pert')
WORKERS = int(os.environ.get('BENCH_WORKERS', max(4, min(40, (os.cpu_count() or 8) - 4))))
DATASETS = os.environ.get('BENCH_DS', 'eth,hotel,univ,zara1,zara2').split(',')
DRAWS = int(os.environ.get('BENCH_DRAWS', 10))
OUT = os.path.join(WORK, 'out')
T0 = time.time()

GH = 'https://github.com/'
PINNED = {  # repo: (url, commit, data root inside the repo)
    'Social-STGCNN': ('abduallahmohamed/Social-STGCNN', '333d3a57b4d2705e129b21aefefa09c79b2b9ae1', 'datasets'),
    'Social-Implicit': ('abduallahmohamed/Social-Implicit', '02d6a1483d60267acbe7e8caf229f96d6d4921ba', 'datasets'),
    'NPSN': ('InhwanBae/NPSN', '9d2fca316feb6b1235ace0184a0a4d8838dd5436', 'dataset'),
    'DMRGCN': ('InhwanBae/DMRGCN', '5fde49848c1d6adeb6c944957c96a562f9b94524', 'datasets'),
    'GraphTERN': ('InhwanBae/GraphTERN', '00faa3befa07ffc4f7b77a8c22f806db05400767', 'datasets'),
    'EqMotion': ('MediaBrain-SJTU/EqMotion', '5aec2e0b61c511fa93a24138dd90da59a089084b', 'eth_ucy/data'),
    'EigenTrajectory': ('InhwanBae/EigenTrajectory', 'f2f8fc3fa58721a83c60bb407319c6cecd154dfc', 'datasets'),
    'SingularTrajectory': ('InhwanBae/SingularTrajectory', 'f7855881976276787bc16d951a9259d1cd50c101', 'datasets'),
}
RELEASES = [('EigenTrajectory', f'{GH}InhwanBae/EigenTrajectory/releases/download/v1.0/EigenTrajectory-{b}-pretrained.zip')
            for b in ('STGCNN', 'SGCN', 'DMRGCN', 'GPGraph-STGCNN', 'GPGraph-SGCN', 'Graph-TERN', 'Implicit', 'PECNet',
                      'LBEBM', 'AgentFormer')]
RELEASES += [('SingularTrajectory', f'{GH}InhwanBae/SingularTrajectory/releases/download/v1.0/'
                                    'SingularTrajectory-stochastic-pretrained.zip'),
             ('SingularTrajectory', f'{GH}InhwanBae/SingularTrajectory/releases/download/v0.1/datasets.zip'),
             ('SingularTrajectory', f'{GH}InhwanBae/SingularTrajectory/releases/download/v0.1/'
                                    'datasets-imageextension.zip')]  # homography + vector fields
EQ_TEST = {'eth': ['biwi_eth'], 'hotel': ['biwi_hotel'], 'univ': ['students001', 'students003'],
           'zara1': ['crowds_zara01'], 'zara2': ['crowds_zara02']}

sys.path.insert(0, HERE)
from adapters import MODELS  # noqa: E402  (also patches torch.load; harmless here)


def log(*a):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)


def setup():
    os.makedirs(WORK, exist_ok=True)
    if os.environ.get('BENCH_OFFLINE'):                                   # Kaggle: repositories shipped inside the kernel
        json.dump({k: dict(url=GH + v[0], commit=v[1]) for k, v in PINNED.items() if os.path.isdir(os.path.join(REPOS, k))},
                  open(os.path.join(WORK, 'commits.json'), 'w'), indent=1)
        log('offline: using shipped repositories', sorted(os.listdir(REPOS)))
        return
    os.makedirs(REPOS, exist_ok=True)
    for name, (url, commit, _) in PINNED.items():
        d = os.path.join(REPOS, name)
        if not os.path.isdir(os.path.join(d, '.git')):
            subprocess.run(['git', 'clone', '-q', GH + url, d], check=True)
        subprocess.run(['git', '-C', d, 'checkout', '-q', commit], check=True)
    for repo, url in RELEASES:
        flag = os.path.join(REPOS, repo, '.got_' + os.path.basename(url))
        if not os.path.exists(flag):
            z = os.path.join(REPOS, repo, 'dl.zip')
            urllib.request.urlretrieve(url, z)
            zipfile.ZipFile(z).extractall(os.path.join(REPOS, repo))
            os.remove(z)
            open(flag, 'w').close()
    os.makedirs(WORK, exist_ok=True)
    json.dump({k: dict(url=GH + v[0], commit=v[1]) for k, v in PINNED.items()} | {'releases': [u for _, u in RELEASES]},
              open(os.path.join(WORK, 'commits.json'), 'w'), indent=1)
    log('repositories ready')


def remap_file(path, draw):
    """Random bijection of the pedestrian IDs (2nd column) of one file; same draw + file name -> same bijection.
    Rows are rewritten sorted by (frame, new ID), so loaders that take agents in row order (AgentFormer-style) are
    re-indexed too; loaders that sort IDs are unaffected by the row order."""
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    delim = '\t' if '\t' in lines[0] else ' '
    rows = [l.strip().split(delim) if delim == '\t' else l.split() for l in lines]
    ids = sorted({r[1] for r in rows}, key=float)
    rng = np.random.default_rng([draw, zlib.crc32(os.path.basename(path).encode())])
    new = dict(zip(ids, rng.permutation(ids).tolist()))
    for r in rows:
        r[1] = new[r[1]]
    rows.sort(key=lambda r: (float(r[0]), float(r[1])))
    with open(path, 'w') as f:
        f.write(''.join(delim.join(r) + '\n' for r in rows))


def test_files(repo, root, ds):
    if repo == 'EqMotion':
        return [os.path.join(root, ds, n + '.txt') for n in EQ_TEST[ds]]
    d = os.path.join(root, ds, 'test')
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.txt')]


def data_root(repo, cond):
    rel = PINNED[repo][2]
    return os.path.join(REPOS, repo, rel) if cond == 'id' else os.path.join(PERT, repo, cond, rel)


def perturb(repos):
    for repo in repos:
        for r in range(1, DRAWS + 1):
            dst = data_root(repo, f'r{r}')
            if os.path.exists(dst + '.done'):
                continue
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(data_root(repo, 'id'), dst)
            for ds in DATASETS:
                for f in test_files(repo, dst, ds):
                    remap_file(f, r)
            open(dst + '.done', 'w').close()
    log('re-indexed test sets ready')


def main():
    only = [m for m in os.environ.get('BENCH_ONLY', '').split(',') if m] or list(MODELS)
    setup()
    perturb(sorted({MODELS[m][0] for m in only}))
    conds = ['id'] + [f'r{r}' for r in range(1, DRAWS + 1)]
    jobs = [(m, ds, c) for ds in sorted(DATASETS, key=lambda d: d != 'univ') for m in only for c in conds]
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, j[0], j[1], j[2] + '.json'))]
    log(f'jobs: {len(jobs)}, to run: {len(todo)}, parallel: {WORKERS}')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    running, failed, beat = {}, [], time.time()
    os.makedirs(os.path.join(WORK, 'logs'), exist_ok=True)
    while todo or running:
        for j, p in list(running.items()):
            if p.poll() is not None:
                del running[j]
                if p.returncode:
                    failed.append(j)
                    log('FAILED', *j, 'rc', p.returncode)
        while todo and len(running) < WORKERS:
            m, ds, c = j = todo.pop(0)
            lf = open(os.path.join(WORK, 'logs', f'{m}_{ds}_{c}.txt'), 'w')
            running[j] = subprocess.Popen([sys.executable, os.path.join(HERE, 'job.py'), '--model', m, '--ds', ds,
                                           '--cond', c, '--root', data_root(MODELS[m][0], c), '--out', OUT],
                                          env=env, stdout=lf, stderr=subprocess.STDOUT)
        if time.time() - beat > 300:
            beat = time.time()
            log(f'running {len(running)}, left {len(todo)}, failed {len(failed)}')
        time.sleep(2)
    log('all jobs finished; failed:', len(failed))
    json.dump(failed, open(os.path.join(WORK, 'failed.json'), 'w'), indent=1)
    subprocess.run([sys.executable, os.path.join(HERE, 'bench_a_analyze.py'), OUT,
                    os.path.join(WORK, 'results')], check=False)


if __name__ == '__main__':
    main()
