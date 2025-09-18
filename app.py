import os, json, yaml, numpy as np, cv2, shutil
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import pandas as pd

# --------------------------
# Config
# --------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

with open(CFG["names_path"], "r", encoding="utf-8") as f:
    names_gallery = json.load(f)

# --------------------------
# Modelo
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

def detectar_cor(np_img):
    """Detecta cor predominante em nome legível."""
    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:,:,0])
    if 35 <= h_mean <= 85:
        return "verde"
    elif 10 <= h_mean < 35:
        return "amarelo/marrom"
    elif 85 < h_mean <= 135:
        return "azul"
    elif 135 < h_mean <= 170:
        return "roxo"
    else:
        return "vermelho/rosa"

def detectar_tracos(np_img):
    """Heurística: se variância da luminância > limite → pedra tem listras."""
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    return "com listras" if variance > 500 else "uniforme"

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/process", methods=["POST"])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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

    os.makedirs("static/crops", exist_ok=True)
    results_list = []

    for i, (x, y, w, h) in enumerate(boxes, 1):
        crop = img_cv[y:y+h, x:x+w]
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        crop_filename = f"stone_{i}.jpg"
        crop_path = os.path.join("static/crops", crop_filename)
        cv2.imwrite(crop_path, crop)

        q_resnet = embed_resnet(pil_crop)
        q_color, q_shape = embed_color_shape(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        cor_detectada = detectar_cor(crop)
        tracos = detectar_tracos(crop)

        distances = []
        for item in names_gallery:
            if item["label_manual"] != cor_detectada: continue
            d = compute_distance(q_resnet, q_color, q_shape, item)
            distances.append((item["file"], item["label_manual"], d))

        if distances:
            distances.sort(key=lambda x: x[2])
            best_file, best_label, best_dist = distances[0]
            confidence = max(0, 1 - best_dist)
        else:
            best_file, best_label, best_dist, confidence = "N/A", cor_detectada, 999, 0

        results_list.append({
            "id": i,
            "type": cor_detectada,
            "pattern": tracos,
            "best_match_file": best_file,
            "confidence": confidence,
            "distance": round(best_dist, 4),
            "width": w,
            "height": h,
            "area": w * h,
            "crop_url": f"/static/crops/{crop_filename}"
        })

        color = (34, 139, 34) if confidence > 0.7 else (0, 165, 255)
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_cv, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    os.makedirs("static", exist_ok=True)
    annotated_filename = f"annotated_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.jpg"
    annotated_path = os.path.join("static", annotated_filename)
    cv2.imwrite(annotated_path, img_cv)

    if results_list:
        df = pd.DataFrame(results_list)
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/report.csv", index=False, encoding="utf-8")

    # ---- resumo inteligente ----
    summary = {
        "total_pedras": len(results_list),
        "groups": []
    }
    df = pd.DataFrame(results_list)
    for group_name, group_data in df.groupby("type"):
        summary["groups"].append({
            "group": group_name,
            "total": len(group_data),
            "avg_area": round(group_data["area"].mean(), 2),
            "patterns": group_data["pattern"].value_counts().to_dict()
        })

    return jsonify({
        "annotated_image_url": annotated_path,
        "stones": results_list,
        "summary": summary
    })

@app.route("/download_csv")
def download_csv():
    return send_file("outputs/report.csv", as_attachment=True, download_name="report.csv")

@app.route("/export_images")
def export_images():
    from shutil import make_archive
    if not os.path.exists("static/crops"):
        return jsonify({"error": "Nenhuma imagem encontrada"}), 400
    zip_path = "outputs/exported"
    make_archive(zip_path, 'zip', "static/crops")
    return send_file(f"{zip_path}.zip", as_attachment=True, download_name="crops.zip")

@app.route("/clear_outputs", methods=["POST"])
def clear_outputs():
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    if os.path.exists("static/crops"):
        shutil.rmtree("static/crops")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("static/crops", exist_ok=True)
    return jsonify({"status": "ok", "message": "Outputs limpos"})

if __name__ == "__main__":
    app.run(debug=True)
