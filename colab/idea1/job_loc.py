"""One Idea-1 locality job: run one model on one dataset and one scene-composition condition ('id', 'idrep', or 'c<k>').

Writes <out>/<name>/<ds>/<cond>.npz (per-scene packed arrays: sizes, key, pin; plus obs, ade, fde on 'id') and <cond>.json.
"""
import argparse, os, sys, time, json, random, hashlib
import numpy as np


def pkeys(obs, gt):
    """64-bit pedestrian-window key from (obs, gt) rounded to 1 mm."""
    mm = np.round(np.concatenate([obs, gt], 1).reshape(len(obs), -1) * 1000).astype(np.int64)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), 'little', signed=True)
                     for r in mm], dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--ds', required=True)
    p.add_argument('--cond', required=True)
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

    here = os.path.dirname(os.path.abspath(__file__))
    bench_a = os.path.normpath(os.path.join(here, '..', 'bench_a'))
    sys.path[:0] = [bench_a, here]

    import torch
    import adapters

    def seeded(seed=0):
        adapters.SEED = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    mname = a.name or a.model
    out = os.path.abspath(os.path.join(a.out, mname, a.ds))
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    run = adapters.MODELS[a.model][1](a.ds, os.path.abspath(a.root))
    load_s = time.time() - t0

    t1 = time.time()
    seeded(0)
    use_pin = a.model in adapters.NEEDS_PIN

    sizes, K_all, P_all, O_all, A_all, F_all = [], [], [], [], [], []
    ade_all_win, fde_all_win = [], []

    with torch.no_grad():
        for obs, gt, s, mu in run(a.K, use_pin):
            assert s.shape[0] == a.K and s.shape[1:] == gt.shape and obs.shape[1:] == (8, 2), (s.shape, gt.shape, obs.shape)
            n = len(obs)
            if a.cond == 'id':
                err = (s - gt).norm(dim=-1)  # K, n, 12
                ade_w = err.mean(-1).min(0)[0].numpy().astype(np.float32)
                fde_w = err[..., -1].min(0)[0].numpy().astype(np.float32)
                ade_all_win.append(ade_w); fde_all_win.append(fde_w)
            if n < 2:
                continue
            obs_np, gt_np = obs.numpy(), gt.numpy()
            pin_np = (mu if mu is not None else s.mean(0)).numpy().astype(np.float32)
            sizes.append(n)
            K_all.append(pkeys(obs_np, gt_np))
            P_all.append(pin_np)
            if a.cond == 'id':
                O_all.append(obs_np.astype(np.float32))
                A_all.append(ade_w)
                F_all.append(fde_w)

    payload = dict(
        sizes=np.array(sizes, dtype=np.int32),
        key=np.concatenate(K_all) if K_all else np.zeros(0, dtype=np.int64),
        pin=np.concatenate(P_all) if P_all else np.zeros((0, 12, 2), dtype=np.float32),
    )
    if a.cond == 'id':
        payload['obs'] = np.concatenate(O_all) if O_all else np.zeros((0, 8, 2), dtype=np.float32)
        payload['ade'] = np.concatenate(A_all) if A_all else np.zeros(0, dtype=np.float32)
        payload['fde'] = np.concatenate(F_all) if F_all else np.zeros(0, dtype=np.float32)

    np.savez_compressed(os.path.join(out, f'{a.cond}.npz'), **payload)
    meta = dict(
        model=mname, base_model=a.model, variant=a.variant or 'O', ds=a.ds, cond=a.cond,
        n_scenes=len(sizes), n_peds=int(sum(sizes)),
        ade_all=float(np.concatenate(ade_all_win).mean()) if ade_all_win else None,
        fde_all=float(np.concatenate(fde_all_win).mean()) if fde_all_win else None,
        load_s=round(load_s, 2), eval_s=round(time.time() - t1, 2),
    )
    json.dump(meta, open(os.path.join(out, f'{a.cond}.json'), 'w'), indent=1)
    print(mname, a.ds, a.cond, f"scenes={meta['n_scenes']} peds={meta['n_peds']} {meta['eval_s']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
