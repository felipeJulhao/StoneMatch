import os, json, yaml, numpy as np, cv2
from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from collections import defaultdict
import pandas as pd

# --------------------------
# Carregar config
# --------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# --------------------------
# Carregar galeria (features já salvas pelo build_index.py)
# --------------------------
with open(CFG["names_path"], "r", encoding="utf-8") as f:
    names_gallery = json.load(f)

# --------------------------
# Modelo de embeddings (ResNet50)
# --------------------------
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

def embed_resnet(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        v = backbone(x).squeeze(0).cpu().numpy().astype("float32")
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def embed_color_shape(np_img):
    # Cor (histograma HSV)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    # Forma (Momentos de Hu)
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

# Carregar pesos do config
alpha = CFG["weights"]["alpha"]
beta = CFG["weights"]["beta"]
gamma = CFG["weights"]["gamma"]

def compute_distance(q_resnet, q_color, q_shape, item, alpha=alpha, beta=beta, gamma=gamma):
    g_resnet = np.array(item["resnet"], dtype="float32")
    g_color = np.array(item["color"], dtype="float32")
    g_shape = np.array(item["shape"], dtype="float32")

    # Distância ResNet (cosseno)
    d_resnet = 1 - np.dot(q_resnet, g_resnet) / (np.linalg.norm(q_resnet)*np.linalg.norm(g_resnet)+1e-9)
    # Distância Cor (Bhattacharyya)
    d_color = cv2.compareHist(q_color.astype("float32"), g_color.astype("float32"), cv2.HISTCMP_BHATTACHARYYA)
    # Distância Forma (Euclidiana)
    d_shape = np.linalg.norm(q_shape - g_shape)

    return alpha*d_color + beta*d_shape + gamma*d_resnet

def detectar_grupo(np_img):
    """Decide se a pedra é 'verdes' ou 'marrons' pela cor predominante (HSV)."""
    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:,:,0])  # matiz médio
    if 35 <= h_mean <= 85:  # faixa aproximada de verde
        return "verdes"
    else:
        return "marrons"

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    annotated_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            # Carregar imagem enviada
            img = Image.open(file.stream).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Detectar pedras via OpenCV
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (CFG["blur_ksize"], CFG["blur_ksize"]), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w >= CFG["min_width"] and h >= CFG["min_height"]:
                    boxes.append((x, y, w, h))
            boxes.sort(key=lambda b: (b[1] // 100, b[0]))

            clusters_out = defaultdict(list)

            for i, (x, y, w, h) in enumerate(boxes, 1):
                crop = img_cv[y:y+h, x:x+w]
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                # Extrair features
                q_resnet = embed_resnet(pil_crop)
                q_color, q_shape = embed_color_shape(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                # Detectar grupo (verde ou marrom)
                grupo = detectar_grupo(crop)

                # Comparar só com a galeria desse grupo
                distances = []
                for item in names_gallery:
                    if item["label_manual"] != grupo:
                        continue
                    d = compute_distance(q_resnet, q_color, q_shape, item)
                    distances.append((item["file"], item["label_manual"], d))

                if distances:
                    distances.sort(key=lambda x: x[2])
                    best_file, best_label, best_dist = distances[0]
                    top5 = [f"{f}:{d:.3f}" for f,_,d in distances[:5]]
                else:
                    best_file, best_label, best_dist, top5 = "N/A", grupo, 999, []

                clusters_out[grupo].append({
                    "crop": f"crop_{i:03d}.jpg",
                    "best_match": best_file,
                    "label_manual": best_label,
                    "distance": round(best_dist, 3),
                    "top5": ";".join(top5)
                })

                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_cv, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Salvar imagem anotada
            os.makedirs("static", exist_ok=True)
            annotated_path = "static/annotated.jpg"
            cv2.imwrite(annotated_path, img_cv)

            # Salvar resultados
            rows = []
            for cid, items in clusters_out.items():
                for it in items:
                    rows.append(it)
            results = pd.DataFrame(rows)

            os.makedirs("outputs", exist_ok=True)
            results.to_csv("outputs/report.csv", index=False, encoding="utf-8")

    return render_template("index.html", results=results, annotated=annotated_path)

@app.route("/download_csv")
def download_csv():
    return send_file("outputs/report.csv", as_attachment=True, download_name="report.csv")

if __name__ == "__main__":
    app.run(debug=True)
