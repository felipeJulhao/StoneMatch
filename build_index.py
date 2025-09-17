import os, json, yaml, numpy as np, cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# --------------------------
# Carregar config
# --------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# --------------------------
# Definir device
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --------------------------
# Modelo backbone (ResNet50)
# --------------------------
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone.fc = nn.Identity()
backbone.eval().to(device)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def embed_resnet(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        v = backbone(x).squeeze(0).cpu().numpy().astype("float32")
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def embed_color_shape(np_img):
    # Cor (histograma HSV normalizado)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    # Forma (Momentos de Hu normalizados)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu)+1e-9)  # log scale
    else:
        hu = np.zeros(7)
    return hist, hu

# --------------------------
# Indexar imagens da galeria
# --------------------------
data = []

for root, _, files in os.walk(CFG["gallery_dir"]):
    for name in sorted(files):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        path = os.path.join(root, name)
        rel_path = os.path.relpath(path, CFG["gallery_dir"])
        label_manual = os.path.dirname(rel_path).split(os.sep)[0] or "none"

        try:
            pil = Image.open(path).convert("RGB")
            np_img = np.array(pil)

            feat_resnet = embed_resnet(pil)
            feat_color, feat_shape = embed_color_shape(np_img)

            data.append({
                "file": rel_path,
                "label_manual": label_manual,
                "resnet": feat_resnet.tolist(),
                "color": feat_color.tolist(),
                "shape": feat_shape.tolist()
            })
        except Exception as e:
            print(f"[ERRO] {path}: {e}")

if not data:
    raise SystemExit("Nenhuma imagem encontrada na galeria.")

# --------------------------
# Salvar index
# --------------------------
os.makedirs("outputs", exist_ok=True)
with open(CFG["names_path"], "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Indexado: {len(data)} imagens")
