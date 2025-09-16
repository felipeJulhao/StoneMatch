import cv2, numpy as np, pickle
from pathlib import Path
from .config import INDEX_DIR
from .features import preprocess_and_features

def _imread(p: Path): return cv2.imread(str(p), cv2.IMREAD_COLOR)

def _load_index():
    with open(INDEX_DIR/"nn.pkl","rb") as f:
        obj = pickle.load(f)
    return obj["nn"], obj["ids"], obj["feats"]

def topk_by_image_path(image_path: str, k: int = 5):
    img = _imread(Path(image_path))
    if img is None: raise FileNotFoundError(image_path)
    _, _, feat = preprocess_and_features(img)
    nn, ids, _ = _load_index()
    k = max(1, min(k, len(ids)))
    dists, idxs = nn.kneighbors([feat], n_neighbors=k)
    scores = 1.0 - dists[0]  # cosine -> similaridade
    return [(ids[i].item(), float(scores[j])) for j,i in enumerate(idxs[0])]
