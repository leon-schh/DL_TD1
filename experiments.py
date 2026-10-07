"""
Label-free experiments that justify the choices made in pipeline.py.

    python experiments.py

Everything is saved in data/cache/report/ (PNG figures + contact sheets)
and the numbers are printed in the terminal. No Caltech-256 label is used.
"""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

import pipeline as P

PHOTO_DIR = Path("data/photos")
REPORT = P.CACHE_DIR / "report"
REPORT.mkdir(parents=True, exist_ok=True)
K_NN = 10

photos = P.get_photos(PHOTO_DIR)
names = [p.name for p in photos]


# ------------------------------------------------------------------ helpers
def knn_idx(X, k=K_NN):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def overlap(a, b):
    """Mean fraction of shared neighbours between two kNN index matrices."""
    return float(np.mean([len(set(x) & set(y)) / a.shape[1] for x, y in zip(a, b)]))


def sheet(paths, cols, thumb=128):
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb - 4, thumb - 4))
        canvas.paste(im, ((i % cols) * thumb + 2, (i // cols) * thumb + 2))
    return canvas


def title(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


# ------------------------------------------- A. encoders: DINOv2 vs ResNet50
title("A. Self-supervised (DINOv2) vs supervised ImageNet (ResNet50)")
encoders = ["dinov2_vits14", "resnet50"]
Zs = {e: P.get_Z(PHOTO_DIR, e) for e in encoders}
fig, ax = plt.subplots(figsize=(6, 4))
for e in encoders:
    labels, k, scores = P.cluster_embeddings(Zs[e])
    print(f"{e:15s} d={Zs[e].shape[1]:5d}  best K={k:2d}  silhouette={scores[k]:.3f}")
    ax.plot(list(scores), list(scores.values()), marker="o", label=e)
ax.set_xlabel("K")
ax.set_ylabel("cosine silhouette")
ax.legend()
fig.tight_layout()
fig.savefig(REPORT / "A_silhouette_vs_K.png", dpi=150)

nn_idx = {e: knn_idx(Zs[e]) for e in encoders}
print(f"kNN overlap@{K_NN} DINOv2 vs ResNet50: "
      f"{overlap(nn_idx[encoders[0]], nn_idx[encoders[1]]):.3f}")

rng = random.Random(0)
queries = rng.sample(range(len(photos)), min(6, len(photos)))
for e in encoders:
    rows = []
    for q in queries:
        rows.append(sheet([photos[q]] + [photos[j] for j in nn_idx[e][q][:5]], cols=6))
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save(REPORT / f"A_neighbours_{e}.png")

# Everything below uses the pipeline's encoder.
Z = Zs[P.ENCODER]

# ----------------------------------------- B. projections: PCA / t-SNE / UMAP
title("B. Dimensionality reduction (trustworthiness & kNN preservation)")
Ys = {}
for m in ["pca", "tsne", "umap"]:
    Ys[m] = P.reduce_dim(Z, m)
    tw = trustworthiness(Z, Ys[m], n_neighbors=K_NN, metric="cosine")
    ov = overlap(knn_idx(Z), knn_idx(Ys[m]))
    print(f"{m:5s} trustworthiness@{K_NN}={tw:.3f}  kNN overlap@{K_NN}={ov:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (m, Y) in zip(axes, Ys.items()):
    ax.scatter(Y[:, 0], Y[:, 1], s=4)
    ax.set_title(m.upper())
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(REPORT / "B_projections.png", dpi=150)

print("UMAP sensitivity (n_neighbors, min_dist):")
for nb in (5, 15, 50):
    for md in (0.0, 0.1, 0.5):
        Y = P.reduce_dim(Z, "umap", n_neighbors=nb, min_dist=md)
        tw = trustworthiness(Z, Y, n_neighbors=K_NN, metric="cosine")
        print(f"  n_neighbors={nb:3d} min_dist={md:.1f}  trustworthiness={tw:.3f}")

# --------------------------------------- C. similarity: cosine vs Euclidean
title("C. Similarity measures")
raw = P.raw_embeddings(PHOTO_DIR)
print(f"raw vector norms: min={np.linalg.norm(raw, axis=1).min():.1f} "
      f"max={np.linalg.norm(raw, axis=1).max():.1f}")
cos_idx = knn_idx(Z)  # Euclidean on L2-normalised == cosine ranking
euc_raw_idx = knn_idx(raw)
cos_raw = np.argsort(-(raw / np.linalg.norm(raw, axis=1, keepdims=True)) @
                     (raw / np.linalg.norm(raw, axis=1, keepdims=True)).T, axis=1)[:, 1:K_NN + 1]
print(f"overlap@{K_NN} cosine vs Euclidean on RAW vectors: {overlap(cos_raw, euc_raw_idx):.3f}")
print(f"overlap@{K_NN} cosine vs Euclidean on NORMALISED vectors: "
      f"{overlap(cos_raw, cos_idx):.3f} (must be ~1.0: same ranking)")

# ----------------------------------- D. which space to cluster: Z or 2D map
title("D. Clustering space (Z vs 2D UMAP) and method (KMeans vs HDBSCAN)")
labels_Z, k, scores = P.cluster_embeddings(Z)
Y2 = Ys["umap"]
labels_Y = KMeans(n_clusters=k, n_init=10, random_state=P.SEED).fit_predict(Y2)
print(f"K={k}")
print(f"KMeans on Z : silhouette (in Z, cosine) = {silhouette_score(Z, labels_Z, metric='cosine'):.3f}")
print(f"KMeans on 2D: silhouette (in Z, cosine) = {silhouette_score(Z, labels_Y, metric='cosine'):.3f}")
print(f"Adjusted Rand index between the two partitions = {adjusted_rand_score(labels_Z, labels_Y):.3f}")

Y10 = P.reduce_dim(Z, "umap", n_components=10)
hdb = HDBSCAN(min_cluster_size=max(5, len(Z) // 100)).fit_predict(Y10)
mask = hdb >= 0
n_hdb = len(set(hdb[mask]))
print(f"HDBSCAN on UMAP-10D: {n_hdb} clusters, {100 * (~mask).mean():.1f}% noise")
if n_hdb > 1:
    print(f"  silhouette (in Z, cosine, non-noise only) = "
          f"{silhouette_score(Z[mask], hdb[mask], metric='cosine'):.3f}")
    print(f"  ARI vs KMeans-on-Z (non-noise) = {adjusted_rand_score(labels_Z[mask], hdb[mask]):.3f}")

# --------------------------------------------- E. qualitative inspection
title("E. Qualitative inspection (contact sheets)")
for g in range(k):
    idx = np.where(labels_Z == g)[0]
    sel = rng.sample(list(idx), min(24, len(idx)))
    sheet([photos[i] for i in sel], cols=8).save(REPORT / f"E_group_{g:02d}_n{len(idx)}.png")

sil = silhouette_samples(Z, labels_Z, metric="cosine")
worst = np.argsort(sil)[:24]
sheet([photos[i] for i in worst], cols=8).save(REPORT / "E_worst_assigned.png")

dist, _ = NearestNeighbors(n_neighbors=K_NN + 1).fit(Z).kneighbors(Z)
isolated = np.argsort(-dist[:, 1:].mean(axis=1))[:24]
sheet([photos[i] for i in isolated], cols=8).save(REPORT / "E_isolated.png")
print(f"Contact sheets saved in {REPORT}/")