"""Phase-4(alt) Module B: train a dense embedding of the sparse Gin Rummy obs from
the LLM-judged similarity dataset (obs1, obs2, score in 0-5).

Embedder f: R^208 -> R^EMB_DIM (an MLP W1 sigma(W2 sigma(W3 x +b)+b)+b). We fit
cosine(f(o1), f(o2)) to the LLM similarity mapped to [-1,1]. Negative (low-score)
pairs push dissimilar states apart, preventing collapse. Quality = correlation of
predicted cosine vs the LLM score on a held-out split.

    python sweep/embed_train.py            # reads sweep/embed/sim_dataset.npz
    EMB_DIM=20 EPOCHS=60 python sweep/embed_train.py
"""
import os, sys, json, glob
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch as th
import torch.nn as nn


def _load_dataset(ds_path):
    """Load one .npz or merge a glob of shards (sim_dataset_shard*.npz)."""
    files = sorted(glob.glob(ds_path)) if any(c in ds_path for c in "*?[") else [ds_path]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        sys.exit(f"no dataset files match {ds_path}")
    O1 = np.concatenate([np.load(f)["obs1"] for f in files])
    O2 = np.concatenate([np.load(f)["obs2"] for f in files])
    raw = np.concatenate([np.load(f)["score"] for f in files]).astype(np.float64)
    print(f"loaded {len(files)} shard(s) -> {len(raw)} pairs", flush=True)
    return O1, O2, raw


class Embedder(nn.Module):
    """f: 208 -> EMB_DIM, MLP with ReLU hiddens (linear output for cosine)."""
    def __init__(self, in_dim=208, emb_dim=20, hidden=(128, 64)):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(dims[-1], emb_dim)]
        self.net = nn.Sequential(*layers)
        self.cfg = dict(in_dim=in_dim, emb_dim=emb_dim, hidden=list(hidden))

    def forward(self, x):
        return self.net(x)


def main():
    ds_path = os.environ.get("DATASET",
                             os.path.join(PROJECT_ROOT, "sweep", "embed", "sim_dataset.npz"))
    emb_dim = int(os.environ.get("EMB_DIM", 20))
    epochs = int(os.environ.get("EPOCHS", 60))
    lr = float(os.environ.get("LR", 1e-3))
    batch = int(os.environ.get("BATCH", 256))
    out = os.environ.get("OUT", os.path.join(PROJECT_ROOT, "sweep", "embed", "embedder.pt"))
    seed = int(os.environ.get("SEED", 0))
    th.manual_seed(seed); np.random.seed(seed)

    o1, o2, raw = _load_dataset(ds_path)
    O1 = th.tensor(o1, dtype=th.float32)
    O2 = th.tensor(o2, dtype=th.float32)
    # DEBIAS: the LLM clusters its raw scores, so rank-bin them into 6 equal-frequency
    # buckets 0-5 (it's the RANK that matters, not the absolute number), then map to a
    # cosine target in [-1,1].
    ranks = np.argsort(np.argsort(raw))
    bins = (ranks * 6 // len(raw)).clip(0, 5)
    tgt = th.tensor(bins, dtype=th.float32) / 5.0 * 2.0 - 1.0
    import collections
    print(f"[debias] raw 0-100 mean={raw.mean():.1f} std={raw.std():.1f} -> 0-5 bin "
          f"counts {dict(sorted(collections.Counter(bins.tolist()).items()))}", flush=True)
    n = len(tgt); idx = np.random.permutation(n); nval = max(1, n // 5)
    vi, ti = idx[:nval], idx[nval:]
    print(f"=== embed-train n={n} emb_dim={emb_dim} epochs={epochs} "
          f"(train {len(ti)} / val {len(vi)}) ===", flush=True)

    model = Embedder(in_dim=O1.shape[1], emb_dim=emb_dim)
    opt = th.optim.Adam(model.parameters(), lr=lr)
    cos = nn.CosineSimilarity(dim=1)

    def predict(o1, o2):
        return cos(model(o1), model(o2))

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

    ti_t = th.tensor(ti)
    for ep in range(epochs):
        model.train(); perm = ti_t[th.randperm(len(ti_t))]
        tot = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            pred = predict(O1[b], O2[b])
            loss = ((pred - tgt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with th.no_grad():
                vp = predict(O1[vi], O2[vi])
                vloss = float(((vp - tgt[vi]) ** 2).mean())
                vcorr = corr(vp, tgt[vi])
            print(f"  ep {ep:3d} train_mse={tot/len(ti):.4f} val_mse={vloss:.4f} "
                  f"val_corr={vcorr:.3f}", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    th.save({"state_dict": model.state_dict(), "cfg": model.cfg}, out)
    with th.no_grad():
        vp = predict(O1[vi], O2[vi]); final_corr = corr(vp, tgt[vi])
    print(f"[done] saved {out} | val_corr={final_corr:.3f}", flush=True)
    json.dump({"val_corr": final_corr, "emb_dim": emb_dim, "n": n,
               "epochs": epochs}, open(out.replace(".pt", "_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
