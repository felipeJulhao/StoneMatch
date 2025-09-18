# ... (todos os seus imports e configurações de modelo continuam os mesmos) ...
import os, json, yaml, numpy as np, cv2
from flask import Flask, render_template, request, jsonify, send_file
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
# Modelo de embeddings (ResNet50) e Funções
# ... (todas as suas funções como embed_resnet, embed_color_shape, compute_distance, etc. continuam aqui, sem alterações)
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
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu)+1e-9)
    else:
        hu = np.zeros(7)
    return hist, hu

alpha = CFG["weights"]["alpha"]
beta = CFG["weights"]["beta"]
gamma = CFG["weights"]["gamma"]

def compute_distance(q_resnet, q_color, q_shape, item, alpha=alpha, beta=beta, gamma=gamma):
    g_resnet = np.array(item["resnet"], dtype="float32")
    g_color = np.array(item["color"], dtype="float32")
    g_shape = np.array(item["shape"], dtype="float32")
    d_resnet = 1 - np.dot(q_resnet, g_resnet)
    d_color = cv2.compareHist(q_color.astype("float32"), g_color.astype("float32"), cv2.HISTCMP_BHATTACHARYYA)
    d_shape = np.linalg.norm(q_shape - g_shape)
    return alpha*d_color + beta*d_shape + gamma*d_resnet

def detectar_grupo(np_img):
    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:,:,0])
    if 35 <= h_mean <= 85:
        return "verdes"
    else:
        return "marrons"

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Esta rota agora APENAS serve a página HTML principal."""
    return render_template("index.html")

@app.route("/api/process", methods=["POST"])
def process_image():
    """NOVA ROTA: Esta é a API que processa a imagem e retorna JSON."""
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ... (lógica de detecção com OpenCV, igual à anterior)
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

    results_list = []
    for i, (x, y, w, h) in enumerate(boxes, 1):
        crop = img_cv[y:y+h, x:x+w]
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        q_resnet = embed_resnet(pil_crop)
        q_color, q_shape = embed_color_shape(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        grupo = detectar_grupo(crop)

        distances = []
        for item in names_gallery:
            if item["label_manual"] != grupo: continue
            d = compute_distance(q_resnet, q_color, q_shape, item)
            distances.append((item["file"], item["label_manual"], d))

        if distances:
            distances.sort(key=lambda x: x[2])
            best_file, best_label, best_dist = distances[0]
            # Convertendo distância para 'confiança' (0 a 1). 
            # Ajuste a fórmula se necessário.
            confidence = max(0, 1 - best_dist)
        else:
            best_file, best_label, best_dist, confidence = "N/A", grupo, 999, 0

        # Montando o dicionário para a API
        results_list.append({
            "id": i,
            "type": best_label, # Ex: 'verdes' ou 'marrons'
            "best_match_file": best_file,
            "confidence": confidence,
            "distance": round(best_dist, 4),
            "width": w, # Em pixels, pode ser convertido para cm se tiver a escala
            "height": h,
            "area": w * h
        })

        # Desenha na imagem
        color = (34, 139, 34) if confidence > 0.7 else (0, 165, 255)
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_cv, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Salvar imagem anotada
    os.makedirs("static", exist_ok=True)
    # Adicionamos um timestamp para evitar problemas de cache do navegador
    annotated_filename = f"annotated_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.jpg"
    annotated_path = os.path.join("static", annotated_filename)
    cv2.imwrite(annotated_path, img_cv)

    # Salvar o report.csv (opcional, mas bom manter)
    if results_list:
        df = pd.DataFrame(results_list)
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/report.csv", index=False, encoding="utf-8")

    # Retorna o JSON para o frontend
    return jsonify({
        "annotated_image_url": annotated_path,
        "stones": results_list
    })

@app.route("/download_csv")
def download_csv():
    return send_file("outputs/report.csv", as_attachment=True, download_name="report.csv")

if __name__ == "__main__":
    app.run(debug=True)