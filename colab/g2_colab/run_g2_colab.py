"""Colab GPU runner for IJCAI 2027 Stream G2 (replacing queued Kaggle GPU #2 `ijcai-osc-mot-g2`):
- Trains full 5-seed GCN & Kinematic TTC-Cone suite (B, K, K0, KR, KN, KC) across all 5 ETH/UCY splits (150 runs)
  + side-by-side TFW & TFC1 seed 0 (10 runs)
  + Stanford Drone Dataset (SDD) 3-seed suite (B, K, K0, KN, KC = 15 runs) + TFW & TFC1 seed 0 (2 runs) = 177 GPU runs.
- Saves every completed run directly into G2_WORK/out on Google Drive (100% resumable across Colab disconnects).
- Runs `src/ijcai_probe.py` automatically at completion to produce `ijcai_k2_summary.json` and `ijcai_k2_summary.md`
  (Locality curves, out_cone_max, Approaching ADE, MOT-Glitch stress test, TTC-OSC 1/2/3-sweep & JADE bounds).
"""
import os, sys, json, time, shutil, zipfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'src'), ROOT]

WORK = os.environ.get('G2_WORK', '/content/g2_work')
OUT = os.path.join(WORK, 'out')
REPOS = os.environ.get('BENCH_REPOS', '/content/repos')
TMP = os.environ.get('G2_TMP', '/content/g2_tmp')
WORKERS = int(os.environ.get('G2_WORKERS', 6))
SMOKE = os.environ.get('G2_SMOKE', '') == '1'
ONLY_DS = [x.strip() for x in os.environ.get('G2_DS', '').split(',') if x.strip()]
T0 = time.time()

ETH = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
GCN_CFGS = {
    'B': ['--arch', 'stgcnn', '--graph', 'A_off', '--txp', '3', '3'],
    'K': ['--arch', 'stgcnn', '--graph', 'A_off', '--txp', '3', '1'],
    'K0': ['--arch', 'stgcnn', '--graph', 'A_eye', '--txp', '3', '1'],
    'KR': ['--arch', 'stgcnn', '--graph', 'A_rad', '--txp', '3', '1'],
    'KN': ['--arch', 'stgcnn', '--graph', 'A_radn', '--txp', '3', '1'],
    'KC': ['--arch', 'stgcnn', '--graph', 'A_cone', '--txp', '3', '1'],
}
TF_CFGS = {
    'TFW': ['--arch', 'tf_world', '--graph', 'A_eye', '--txp', '3', '1'],
    'TFC1': ['--arch', 'tf_cone1', '--graph', 'A_eye', '--txp', '3', '1'],
}


def log(*a):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)


def prepare_data():
    """Ensure ETH/UCY raw data (from Social-STGCNN pinned commit) and SDD (from bundled sdd_eth_format.zip) exist in TMP."""
    os.makedirs(TMP, exist_ok=True)
    # 1. ETH/UCY from Social-STGCNN repo
    stg_dir = os.path.join(REPOS, 'Social-STGCNN')
    if not os.path.isdir(os.path.join(stg_dir, 'datasets', 'eth', 'train')):
        os.makedirs(REPOS, exist_ok=True)
        if not os.path.isdir(os.path.join(stg_dir, '.git')):
            subprocess.run(['git', 'clone', '-q', 'https://github.com/abduallahmohamed/Social-STGCNN', stg_dir], check=True)
        subprocess.run(['git', '-C', stg_dir, 'checkout', '-q', '333d3a57b4d2705e129b21aefefa09c79b2b9ae1'], check=True)
    eth_root = os.path.join(stg_dir, 'datasets')

    # 2. SDD from bundled data/ijcai_g2/sdd_eth_format.zip
    sdd_root = os.path.join(TMP, 'sdd_eth')
    z_sdd = os.path.join(ROOT, 'data', 'ijcai_g2', 'sdd_eth_format.zip')
    if not os.path.isdir(os.path.join(sdd_root, 'sdd', 'train')) and os.path.exists(z_sdd):
        os.makedirs(sdd_root, exist_ok=True)
        with zipfile.ZipFile(z_sdd) as z:
            z.extractall(sdd_root)
    log('data ready | eth_root:', eth_root, '| sdd ready:', os.path.isdir(os.path.join(sdd_root, 'sdd', 'train')))
    return eth_root, sdd_root


def build_jobs():
    if SMOKE:
        return [
            dict(name='zara1_IJCAI-KN_s0', ds='zara1', args=GCN_CFGS['KN'] + ['--seed', '0', '--epochs', '1']),
            dict(name='zara1_IJCAI-KC_s0', ds='zara1', args=GCN_CFGS['KC'] + ['--seed', '0', '--epochs', '1']),
            dict(name='zara1_IJCAI-TFW_s0', ds='zara1', args=TF_CFGS['TFW'] + ['--seed', '0', '--epochs', '1']),
            dict(name='zara1_IJCAI-TFC1_s0', ds='zara1', args=TF_CFGS['TFC1'] + ['--seed', '0', '--epochs', '1']),
            dict(name='sdd_IJCAI-KC_s0', ds='sdd', args=GCN_CFGS['KC'] + ['--seed', '0', '--epochs', '1']),
            dict(name='sdd_IJCAI-TFC1_s0', ds='sdd', args=TF_CFGS['TFC1'] + ['--seed', '0', '--epochs', '1']),
        ]
    jobs = [dict(name=f'{ds}_IJCAI-{c}_s{s}', ds=ds, args=GCN_CFGS[c] + ['--seed', str(s)])
            for s in range(5) for ds in ETH for c in GCN_CFGS]
    jobs += [dict(name=f'{ds}_IJCAI-{c}_s0', ds=ds, args=TF_CFGS[c] + ['--seed', '0'])
             for ds in ETH for c in TF_CFGS]
    jobs += [dict(name=f'sdd_IJCAI-{c}_s{s}', ds='sdd', args=GCN_CFGS[c] + ['--seed', str(s)])
             for s in range(3) for c in ('B', 'K', 'K0', 'KN', 'KC')]
    jobs += [dict(name=f'sdd_IJCAI-{c}_s0', ds='sdd', args=TF_CFGS[c] + ['--seed', '0'])
             for c in TF_CFGS]
    if ONLY_DS:
        jobs = [j for j in jobs if j['ds'] in ONLY_DS]
    return jobs


def main():
    os.makedirs(OUT, exist_ok=True)
    eth_root, sdd_root = prepare_data()
    cache_dir = os.path.join(TMP, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    env = dict(os.environ, TTC_DATA_ROOT=eth_root, TTC_SDD_ROOT=sdd_root, TTC_CACHE=cache_dir,
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    jobs = build_jobs()
    if not os.path.isdir(os.path.join(sdd_root, 'sdd', 'train')):
        log('WARNING: SDD directory missing -> skipping SDD jobs')
        jobs = [j for j in jobs if j['ds'] != 'sdd']

    # Pre-build dataset .pt caches in parallel so GPU workers never race on cache writes
    splits = sorted({(j['ds'], sp) for j in jobs for sp in ('train', 'val', 'test')})
    procs = []
    for ds, sp in splits:
        if os.path.exists(os.path.join(cache_dir, f'{ds}_{sp}.pt')):
            continue
        while len([p for p in procs if p.poll() is None]) >= min(6, WORKERS):
            time.sleep(1.5)
        procs.append(subprocess.Popen(
            [sys.executable, '-c', f"import sys; sys.path.insert(0, 'src'); from common import load_split; "
                                   f"print('{ds}', '{sp}', len(load_split('{ds}', '{sp}', need=())))"],
            cwd=ROOT, env=env))
    for p in procs:
        p.wait()
    log('caches ready for splits:', sorted({d for d, _ in splits}))

    manifest = {j['name']: dict(status='done' if os.path.exists(os.path.join(OUT, j['name'], 'result.json')) else 'pending', **j)
                for j in jobs}

    def save_manifest():
        json.dump(dict(elapsed_min=round((time.time() - T0) / 60, 2), jobs=manifest),
                  open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8'), indent=1)

    queue = [j for j in jobs if manifest[j['name']]['status'] != 'done']
    log(f'total jobs: {len(jobs)}, remaining: {len(queue)}, parallel GPU workers: {WORKERS}')
    save_manifest()

    running, failed, beat = {}, [], time.time()
    while queue or running:
        for name, (p, t_st) in list(running.items()):
            if p.poll() is not None:
                ok = p.returncode == 0 and os.path.exists(os.path.join(OUT, name, 'result.json'))
                mins = round((time.time() - t_st) / 60, 2)
                manifest[name].update(status='done' if ok else f'failed rc={p.returncode}', minutes=mins)
                del running[name]
                if not ok:
                    failed.append(name)
                    log('FAILED', name, 'rc', p.returncode)
                else:
                    log('DONE', name, f'{mins:.1f} min')
                save_manifest()
        while queue and len(running) < WORKERS:
            j = queue.pop(0)
            d = os.path.join(OUT, j['name'])
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, 'src/train_fast.py', '--ds', j['ds'], *j['args'], '--device', 'cuda', '--out', d]
            lf = open(os.path.join(d, 'log.txt'), 'w', encoding='utf-8')
            running[j['name']] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT), time.time())
            manifest[j['name']].update(status='running')
        if time.time() - beat > 120:
            beat = time.time()
            done_n = sum(1 for m in manifest.values() if m['status'] == 'done')
            log(f'heartbeat: done={done_n}/{len(jobs)}, running={len(running)}, queued={len(queue)}, failed={len(failed)}')
        time.sleep(1.5)

    json.dump(failed, open(os.path.join(WORK, 'failed.json'), 'w', encoding='utf-8'), indent=1)
    log('all trainings finished; failed:', len(failed), '-> running ijcai_probe.py...')
    dss = sorted({j['ds'] for j in jobs})
    subprocess.run([sys.executable, 'src/ijcai_probe.py', '--runs', OUT, '--datasets', *dss,
                    '--out', os.path.join(OUT, 'ijcai_k2_summary')], cwd=ROOT, env=env, check=False)
    log('G2 stream complete!')


if __name__ == '__main__':
    main()
