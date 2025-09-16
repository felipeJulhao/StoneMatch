import cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
from .config import RAW_DIR, CLEAN_DIR, INDEX_DIR
from .features import preprocess_and_features

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def _imread(p: Path):
    return cv2.imread(str(p), cv2.IMREAD_COLOR)

def ingest_folder(folder: Path | None = None):
    folder = folder or RAW_DIR
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    ids, feats = [], []
    for p in tqdm(imgs, desc="Ingest"):
        img = _imread(p)
        if img is None: continue
        crop, mask, feat = preprocess_and_features(img)
        out = CLEAN_DIR / (p.stem + "_clean.jpg")
        cv2.imwrite(str(out), crop)
        ids.append(out.name); feats.append(feat)

    if feats:
        feats_arr = np.stack(feats).astype(np.float32)
        ids_arr   = np.array(ids, dtype=object)
    else:
        feats_arr = np.zeros((0,8), np.float32)
        ids_arr   = np.array([], dtype=object)

    np.savez_compressed(INDEX_DIR / "features.npz", ids=ids_arr, feats=feats_arr)
    print(f"[OK] {len(ids_arr)} itens ingeridos → index/features.npz")

def build_nn(metric="cosine"):
    data = np.load(INDEX_DIR/"features.npz", allow_pickle=True)
    ids, feats = data["ids"], data["feats"]
    if len(ids)==0: raise RuntimeError("Sem features para indexar.")
    nn = NearestNeighbors(metric=metric)
    nn.fit(feats)
    with open(INDEX_DIR/"nn.pkl","wb") as f:
        pickle.dump({"nn":nn,"ids":ids,"feats":feats}, f)
    print("[OK] Índice salvo em index/nn.pkl")
