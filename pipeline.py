"""
PhotoMap pipeline.

photographs -> pretrained encoder -> Z (N x d, L2-normalised)
                                      |-> UMAP (2D)      -> Photo Map
                                      |-> cosine kNN     -> similar photos
                                      |-> KMeans + silhouette -> Groups

Defaults can be changed with environment variables (used by experiments.py):
    PHOTOMAP_ENCODER    dinov2_vits14 (default) | dinov2_vitb14 | resnet50
    PHOTOMAP_PROJECTION umap (default) | tsne | pca
"""

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ENCODER = os.environ.get("PHOTOMAP_ENCODER", "dinov2_vits14")
PROJECTION = os.environ.get("PHOTOMAP_PROJECTION", "umap")
SEED = 42

# Range explored when choosing the number of groups (see select_k).
K_MIN, K_MAX = 5, 30

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"

# Flask handles requests in several threads: the (expensive) computations are
# done once, protected by a lock, and then kept in memory.
_CACHE = {}
_LOCK = threading.RLock()


def _memo(key, fn):
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = fn()
        return _CACHE[key]


def _log(msg):
    print(f"[pipeline] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 0. Collection
# ---------------------------------------------------------------------------
def get_photos(photo_dir):
    """Return the sorted list of image paths of the collection."""
    photo_dir = Path(photo_dir)
    if not photo_dir.exists():
        return []
    return [
        p
        for p in sorted(photo_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# 1. Representations (Task 1)
# ---------------------------------------------------------------------------
def _load_encoder(name):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if name.startswith("dinov2"):
        # Output of the model = CLS token (after the final LayerNorm).
        model = torch.hub.load("facebookresearch/dinov2", name)
    elif name == "resnet50":
        # Supervised ImageNet baseline: we drop the classification layer and
        # keep the 2048-d global-average-pooled features.
        import torchvision

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown encoder: {name}")

    return model.eval().to(device), device


def extract_embeddings(photos, encoder=None, batch_size=32):
    """Return the raw (un-normalised) representation matrix, shape (N, d)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    encoder = encoder or ENCODER
    model, device = _load_encoder(encoder)

    # Preprocessing expected by DINOv2 and torchvision ResNets (ImageNet stats).
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    class PhotoDataset(Dataset):
        def __len__(self):
            return len(photos)

        def __getitem__(self, i):
            # convert("RGB"): Caltech-256 contains grayscale images.
            return preprocess(Image.open(photos[i]).convert("RGB"))

    loader = DataLoader(PhotoDataset(), batch_size=batch_size, shuffle=False, num_workers=0)

    chunks = []
    with torch.no_grad():
        for b, batch in enumerate(loader):
            chunks.append(model(batch.to(device)).float().cpu().numpy())
            _log(f"{encoder}: batch {b + 1}/{len(loader)}")

    return np.concatenate(chunks, axis=0).astype(np.float32)


def raw_embeddings(photo_dir, encoder=None):
    """Raw embeddings, cached on disk (row i <-> sorted photo i)."""
    encoder = encoder or ENCODER

    def compute():
        photos = get_photos(photo_dir)
        names = np.array([p.name for p in photos])
        cache_file = CACHE_DIR / f"embeddings_{encoder}.npz"

        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=False)
            if np.array_equal(data["names"], names):
                _log(f"embeddings loaded from {cache_file}")
                return data["Z"]

        Z = extract_embeddings(photos, encoder)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, Z=Z, names=names)
        return Z

    return _memo(("raw", str(Path(photo_dir).resolve()), encoder), compute)


def sanity_checks(Z, names):
    """Basic checks of the representation matrix (Section 3.4)."""
    assert Z.shape[0] == len(names), "one row per photograph expected"
    assert np.isfinite(Z).all(), "NaN / Inf in the embeddings"
    norms = np.linalg.norm(Z, axis=1)
    n_dup = len(Z) - len(np.unique(np.round(Z, 5), axis=0))
    _log(
        f"Z shape={Z.shape} | norms min/mean/max="
        f"{norms.min():.2f}/{norms.mean():.2f}/{norms.max():.2f} | "
        f"duplicated vectors={n_dup}"
    )


def get_Z(photo_dir, encoder=None):
    """L2-normalised representations used by every downstream component."""
    encoder = encoder or ENCODER

    def compute():
        Z = raw_embeddings(photo_dir, encoder)
        sanity_checks(Z, [p.name for p in get_photos(photo_dir)])
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    return _memo(("Z", str(Path(photo_dir).resolve()), encoder), compute)


# ---------------------------------------------------------------------------
# 2. Photo Map (Task 2)
# ---------------------------------------------------------------------------
def reduce_dim(Zn, method=None, n_components=2, seed=SEED, **kwargs):
    """Dimensionality reduction of the normalised embeddings."""
    method = method or PROJECTION
    n = len(Zn)

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=seed).fit_transform(Zn)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = kwargs.get("perplexity", min(30, max(5, (n - 1) / 3)))
        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            metric="cosine",
            init="pca",
            random_state=seed,
        ).fit_transform(Zn)

    if method == "umap":
        import umap

        return umap.UMAP(
            n_components=n_components,
            n_neighbors=min(kwargs.get("n_neighbors", 15), n - 1),
            min_dist=kwargs.get("min_dist", 0.1),
            metric="cosine",
            random_state=seed,
        ).fit_transform(Zn)

    raise ValueError(f"Unknown projection method: {method}")


def minmax(Y, margin=0.05):
    """Rescale each axis to [margin, 1 - margin]."""
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    Y01 = (Y - lo) / np.maximum(hi - lo, 1e-12)
    return margin + (1 - 2 * margin) * Y01


def get_Y(photo_dir):
    def compute():
        return minmax(reduce_dim(get_Z(photo_dir)))

    return _memo(("Y", str(Path(photo_dir).resolve()), ENCODER, PROJECTION), compute)


def project_photos(photo_dir):
    photos = get_photos(photo_dir)
    if not photos:
        return []
    Y = get_Y(photo_dir)
    return [
        {"name": p.name, "x": float(Y[i, 0]), "y": float(Y[i, 1])}
        for i, p in enumerate(photos)
    ]


# ---------------------------------------------------------------------------
# 3. Similar photographs (Task 3)
# ---------------------------------------------------------------------------
def find_neighbours(photo_dir, filename, k=5):
    """
    k nearest photographs of `filename` in the ORIGINAL d-dimensional space.
    Z is L2-normalised, so the dot product is exactly the cosine similarity.
    The displayed score is that cosine similarity (clipped to [0, 1]).
    """
    photos = get_photos(photo_dir)
    names = [p.name for p in photos]
    if filename not in names:
        raise ValueError(filename)

    Z = get_Z(photo_dir)
    q = names.index(filename)

    sims = Z @ Z[q]
    sims[q] = -np.inf  # never return the query itself

    k = min(k, len(names) - 1)
    if k <= 0:
        return []
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]

    return [
        {"name": names[i], "similarity": round(float(max(0.0, sims[i])), 3)}
        for i in top
    ]


# ---------------------------------------------------------------------------
# 4. Groups (Task 4)
# ---------------------------------------------------------------------------
def select_k(Zn, k_min=K_MIN, k_max=K_MAX, seed=SEED):
    """Pick K maximising the cosine silhouette score of a KMeans clustering."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k_max = min(k_max, len(Zn) - 1)
    k_min = min(k_min, k_max)

    scores = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Zn)
        scores[k] = float(silhouette_score(Zn, labels, metric="cosine"))

    return max(scores, key=scores.get), scores


def cluster_embeddings(Zn, k=None, seed=SEED):
    """KMeans on L2-normalised vectors (= spherical KMeans). Returns labels, k, scores."""
    from sklearn.cluster import KMeans

    if len(Zn) < 3:
        return np.zeros(len(Zn), dtype=int), 1, {}

    scores = {}
    if k is None:
        k, scores = select_k(Zn, seed=seed)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Zn)
    return labels, k, scores


def get_labels(photo_dir):
    def compute():
        labels, k, scores = cluster_embeddings(get_Z(photo_dir))
        _log(f"K={k} selected | silhouette per K: "
             + ", ".join(f"{kk}:{s:.3f}" for kk, s in scores.items()))
        return labels

    return _memo(("labels", str(Path(photo_dir).resolve()), ENCODER), compute)


def discover_groups(photo_dir):
    photos = get_photos(photo_dir)
    if not photos:
        return []

    labels = get_labels(photo_dir)
    members = {}
    for photo, label in zip(photos, labels):
        members.setdefault(int(label), []).append(photo.name)

    # Biggest group first, ids/names renumbered accordingly.
    ordered = sorted(members.values(), key=len, reverse=True)
    return [
        {"id": i, "name": f"Group {i + 1}", "photos": names}
        for i, names in enumerate(ordered)
    ]


if __name__ == "__main__":
    # Pre-compute and cache everything: python pipeline.py [photo_dir]
    import sys

    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/photos")
    project_photos(d)
    groups = discover_groups(d)
    _log(f"{len(get_photos(d))} photos, {len(groups)} groups, "
         f"sizes={[len(g['photos']) for g in groups]}")