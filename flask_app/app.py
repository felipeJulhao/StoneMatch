import threading, time, sys, os, base64
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from app.config import RAW_DIR, CLEAN_DIR, INDEX_DIR, PAIRS_CSV
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"

app = Flask(__name__, template_folder="templates", static_folder="static")



@app.route("/")
def index(): return render_template("index.html")


@app.post("/save_image")
def route_save_image():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400

        # Remove o cabeçalho "data:image/png;base64," para decodificar
        header, encoded = data['image'].split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Cria um nome de arquivo único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"upload_{timestamp}.png"

        filepath = UPLOADS_DIR / filename
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        return jsonify({'status': 'ok', 'filename': filename}), 200
    except Exception as e:
        print(f"Erro ao salvar imagem: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    data = request.json["image"]
    image_data = data.split(",")[1]

    with open("captura.png", "wb") as f:
        f.write(base64.b64decode(image_data))

    return jsonify({"status": "ok", "message": "Imagem recebida com sucesso"})

if __name__ == "__main__":
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        cam.release()