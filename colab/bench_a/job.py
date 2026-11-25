"""One A-bench job: one model, one dataset, one data condition ('id' runs seeds 0-4, 'rK' runs seed 0).

Writes <out>/<model>/<ds>/<cond>_s<seed>.npz (per pedestrian-window: key, centroid of the K samples, Gaussian mean when the
head has one, minADE_K, minFDE_K) and <cond>.json when the job is complete.
"""
import argparse, os, sys, time, json, random, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--ds', required=True)
p.add_argument('--cond', required=True)
p.add_argument('--root', required=True)
p.add_argument('--out', required=True)
p.add_argument('--K', type=int, default=20)
p.add_argument('--name', help='output folder instead of the model name (B-retrain: <model>-<variant>-s<seed>)')
a = p.parse_args()
out = os.path.abspath(os.path.join(a.out, a.name or a.model, a.ds))
os.makedirs(out, exist_ok=True)


def keys(obs, gt):
    """Pedestrian-window identity: its own observed and future coordinates in mm (unchanged by re-indexing)."""
    mm = np.round(np.concatenate([obs, gt], 1).reshape(len(obs), -1) * 1000).astype(np.int64)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), 'little', signed=True)
                     for r in mm], dtype=np.int64)


def seeded(seed):
    adapters.SEED = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def collect(pin):
    O, G, C, M, A, F = [], [], [], [], [], []
    with torch.no_grad():
        for obs, gt, s, mu in run(a.K, pin):
            assert s.shape[0] == a.K and s.shape[1:] == gt.shape and obs.shape[1:] == (8, 2), (s.shape, gt.shape, obs.shape)
            err = (s - gt).norm(dim=-1)                                   # K,N,12
            O.append(obs.numpy()); G.append(gt.numpy()); C.append(s.mean(0).numpy())
            A.append(err.mean(-1).min(0)[0].numpy()); F.append(err[..., -1].min(0)[0].numpy())
            if mu is not None:
                M.append(mu.numpy())
    O, G = np.concatenate(O), np.concatenate(G)
    arr = dict(key=keys(O, G), cen=np.concatenate(C).astype(np.float32), ade=np.concatenate(A), fde=np.concatenate(F))
    if M:
        arr['mu'] = np.concatenate(M).astype(np.float32)
    return arr


t0 = time.time()
run = adapters.MODELS[a.model][1](a.ds, os.path.abspath(a.root))
meta = dict(model=a.model, ds=a.ds, cond=a.cond, K=a.K, load_s=time.time() - t0, seeds={})
# released sampler on every seed; seed 0 also gets the noise-pinned output ('pin', amendment 1); identity data runs
# seed 0 a second time ('idrep') to measure numerical non-determinism
runs = [(f'{a.cond}_s{s}', s) for s in ([0, 1, 2, 3, 4] if a.cond == 'id' else [0])]
runs += [('idrep_s0', 0)] if a.cond == 'id' else []
for name, seed in runs:
    t1 = time.time()
    seeded(seed)
    arr = collect(False)
    if seed == 0:
        if a.model in adapters.NEEDS_PIN:
            seeded(0)
            p = collect(True)
            assert (p['key'] == arr['key']).all()
            arr['pin'], arr['ade_pin'] = p['cen'], p['ade']
        else:
            arr['pin'] = arr.get('mu', arr['cen'])
    np.savez_compressed(os.path.join(out, name + '.npz'), **arr)
    meta['seeds'][name] = dict(n=len(arr['key']), ade=float(arr['ade'].mean()), fde=float(arr['fde'].mean()),
                               s=time.time() - t1)
    print(a.model, a.ds, name, meta['seeds'][name], flush=True)
json.dump(meta, open(os.path.join(out, f'{a.cond}.json'), 'w'), indent=1)
