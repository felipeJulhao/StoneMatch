import os, json, yaml, argparse, numpy as np, cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics.pairwise import cosine_distances
import csv
from collections import defaultdict

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="Foto grande para processar")
args = ap.parse_args()

# Config
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# Carregar índice da galeria
emb_gallery = np.load(CFG["index_path"])
with open(CFG["names_path"], "r", encoding="utf-8") as f:
    names_gallery = json.load(f)
with open(CFG["clusters_path"], "r", encoding="utf-8") as f:
    clusters = json.load(f)

# Extrair listas
gallery_files = [item["file"] for item in names_gallery]
gallery_labels = [item["label_manual"] for item in names_gallery]

# Mapa nome -> cluster
name_to_cluster = {}
for cid, lst in clusters.items():
    for n in lst:
        key = n["file"] if isinstance(n, dict) else n
        name_to_cluster[key] = int(cid)

# Modelo de embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone.fc = nn.Identity()
backbone.eval().to(device)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def embed_pil(pil):
    x = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        v = backbone(x).squeeze(0).cpu().numpy().astype("float32")
    v /= (np.linalg.norm(v) + 1e-9)
    return v

# Detecção com OpenCV
img = cv2.imread(args.image)
assert img is not None, "Erro ao carregar a imagem"

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (CFG["blur_ksize"], CFG["blur_ksize"]), 0)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w >= CFG["min_width"] and h >= CFG["min_height"]:
        boxes.append((x, y, w, h))
boxes.sort(key=lambda b: (b[1] // 100, b[0]))

# Saídas
os.makedirs("outputs/detections", exist_ok=True)
ann = img.copy()

# Agrupador por cluster
clusters_out = defaultdict(list)

# Processar cada pedra detectada
for i, (x, y, w, h) in enumerate(boxes, 1):
    crop = img[y:y+h, x:x+w]
    crop_p = f"outputs/detections/crop_{i:03d}.jpg"
    cv2.imwrite(crop_p, crop)

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    q = embed_pil(pil).reshape(1, -1)

    dist = cosine_distances(q, emb_gallery).flatten()
    order = np.argsort(dist)

    best_idx = order[0]
    best_file = gallery_files[best_idx]
    best_label = gallery_labels[best_idx]
    best_dist = float(dist[best_idx])

    cluster = name_to_cluster.get(best_file, -1)
    top5 = [f"{gallery_files[j]}:{dist[j]:.3f}" for j in order[:5]]

    clusters_out[cluster].append([
        os.path.basename(crop_p),
        cluster,
        best_file,
        best_label,
        f"{best_dist:.3f}",
        ";".join(top5)
    ])

    cv2.rectangle(ann, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(ann, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Salvar imagem anotada
cv2.imwrite("outputs/annotated.jpg", ann)

# Salvar relatório agrupado
report_path = "outputs/report.csv"
with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for cid, items in sorted(clusters_out.items(), key=lambda kv: kv[0]):
        writer.writerow([f"# Cluster {cid}"])
        writer.writerow(["crop_file", "assigned_cluster", "best_match", "label_manual", "distance", "top5_matches"])
        writer.writerows(items)
        writer.writerow([])  # linha em branco entre blocos

print("Recortes detectados:", len(boxes))
print("Saídas:")
print("-", report_path)
print("- outputs/annotated.jpg")
print("- outputs/detections/")
