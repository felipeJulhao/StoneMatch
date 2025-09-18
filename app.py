import os, json, yaml, numpy as np, cv2
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import pandas as pd
from collections import defaultdict
import shutil, zipfile
from sklearn.cluster import KMeans

# --------------------------
# Config
# --------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

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

# --------------------------
# Detecta cor principal e características extras
# --------------------------
def detectar_grupo(np_img):
    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)

    if s_mean < 40 and v_mean > 200: return "brancas"
    if s_mean < 40 and v_mean < 80: return "pretas"
    if 35 <= h_mean <= 85: return "verdes"
    if 10 <= h_mean <= 25: return "amarelas"
    if 0 <= h_mean <= 10 or h_mean >= 170: return "vermelhas"
    if 100 <= h_mean <= 140: return "azuis"
    if 15 <= h_mean <= 30 and s_mean > 50: return "marrons"
    return "indefinidas"

def extrair_caracteristicas(np_img):
    """Retorna cores dominantes e info de listras"""
    img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img_reshape = img_rgb.reshape((-1,3))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(img_reshape)
    cores = kmeans.cluster_centers_.astype(int)

    cores_legiveis = [f"RGB({r},{g},{b})" for r,g,b in cores]

    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    densidade_bordas = np.sum(edges > 0) / edges.size
    listrado = densidade_bordas > 0.08

    return {
        "cores_dominantes": cores_legiveis,
        "tem_listras": bool(listrado)
    }

# --------------------------
# Flask
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

    boxes = [(x,y,w,h) for c in cnts for x,y,w,h in [cv2.boundingRect(c)] if w>=CFG["min_width"] and h>=CFG["min_height"]]
    boxes.sort(key=lambda b: (b[1]//100, b[0]))

    os.makedirs("static/crops", exist_ok=True)

    results_list, group_stats = [], defaultdict(list)

    for i, (x,y,w,h) in enumerate(boxes, 1):
        crop = img_cv[y:y+h, x:x+w]
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_path = f"static/crops/crop_{i}.jpg"
        cv2.imwrite(crop_path, crop)

        grupo = detectar_grupo(crop)
        extras = extrair_caracteristicas(crop)

        # Confiança e distância não fazem mais sentido aqui
        confidence = 0.75  
        distance = 0.0     

        results_list.append({
            "id": i,
            "type": grupo,
            "confidence": round(confidence, 3),
            "distance": round(distance, 4),
            "width": w,
            "height": h,
            "area": w*h,
            "crop_url": crop_path,
            "cores_dominantes": extras["cores_dominantes"],
            "tem_listras": extras["tem_listras"]
        })

        group_stats[grupo].append(w*h)

        color = (34,139,34) if confidence>=0.6 else (0,0,255)
        cv2.rectangle(img_cv, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img_cv, f"{i}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    annotated_path = f"static/annotated_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(annotated_path, img_cv)

    # ---- Exportar CSV só com os campos reais
    df = pd.DataFrame(results_list, columns=[
        "id","type","confidence","distance","width","height","area",
        "crop_url","cores_dominantes","tem_listras"
    ])
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/report.csv", index=False, encoding="utf-8")

    summary = {
        "total_pedras": len(results_list),
        "groups": [{"group": g, "total": len(v), "avg_area": round(np.mean(v), 2)} for g,v in group_stats.items()]
    }

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
    zip_path = "outputs/images_export.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for folder in ["static/crops"]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    zipf.write(os.path.join(folder, file))
    return send_file(zip_path, as_attachment=True, download_name="images_export.zip")

@app.route("/clear_outputs", methods=["POST"])
def clear_outputs():
    for folder in ["static/crops", "outputs"]:
        if os.path.exists(folder): shutil.rmtree(folder)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
