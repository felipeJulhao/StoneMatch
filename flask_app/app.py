import threading, time, sys
from pathlib import Path
from typing import List, Tuple
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2

# garantir import do pacote app/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from app.config import RAW_DIR, CLEAN_DIR, INDEX_DIR, PAIRS_CSV
from app.indexer import ingest_folder, build_nn
from app.match import topk_by_image_path

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- câmera ----------
# ... imports já existentes ...
import cv2, time, threading

class Camera:
    def __init__(self, prefer_width=1280, prefer_height=720):
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.cap = None
        self.info = {"opened": False, "index": None, "backend": None, "error": None}

        # tente detectar automaticamente
        self._autodetect(prefer_width, prefer_height)

        # se abriu, inicia thread de leitura
        if self.cap is not None and self.cap.isOpened():
            self.running = True
            threading.Thread(target=self._reader, daemon=True).start()

    def _try_open(self, index, backend_flag, prefer_width, prefer_height):
        try:
            index = 1
            cap = cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
            if not cap or not cap.isOpened():
                if cap: cap.release()
                return None

            # configura resolução desejada (nem toda câmera aceita)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, prefer_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, prefer_height)

            # warm-up: tenta ler alguns frames
            ok = False
            for _ in range(15):
                ret, fr = cap.read()
                if ret and fr is not None:
                    ok = True
                    break
                time.sleep(0.03)

            if not ok:
                cap.release()
                return None

            return cap
        except Exception:
            return None

    def _autodetect(self, prefer_width, prefer_height):
        # ordens de tentativa: índices 0..5, backends variados
        indices = list(range(0, 6))
        backends = [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("CAP_MSMF",  cv2.CAP_MSMF),
            ("DEFAULT",   None),
        ]

        for idx in indices:
            for name, flag in backends:
                cap = self._try_open(idx, flag, prefer_width, prefer_height)
                if cap is not None:
                    self.cap = cap
                    self.info.update({"opened": True, "index": idx, "backend": name, "error": None})
                    return

        self.info.update({"opened": False, "error": "Nenhum dispositivo de câmera retornou frames."})

    def _reader(self):
        while self.running:
            ret, fr = self.cap.read()
            if ret and fr is not None:
                with self.lock:
                    self.frame = fr
            else:
                time.sleep(0.02)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.05)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

# ===== crie a instância usando autodetect =====
cam = Camera()

# ===== adicione esta rota de diagnóstico =====
@app.get("/cam_status")
def cam_status():
    return jsonify(cam.info)


def gen_mjpeg():
    while True:
        fr = cam.get()
        if fr is None: time.sleep(0.02); continue
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ---------- rotas ----------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/ingest")
def route_ingest():
    ingest_folder(RAW_DIR); build_nn()
    return jsonify({"status":"ok"})

@app.post("/snapshot_match")
def snapshot_match():
    if not (INDEX_DIR/"nn.pkl").exists():
        return jsonify({"error":"Índice não encontrado. Rode 'Processar & Indexar'."}), 400
    fr = cam.get()
    if fr is None: return jsonify({"error":"Sem frame da câmera."}), 500
    tmp = ROOT_DIR/"data"/"snapshot.jpg"; tmp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp), fr)
    k = int(request.json.get("k",5))
    res: List[Tuple[str,float]] = topk_by_image_path(str(tmp), k=k)
    out = [{"id":rid, "score":float(sc), "url":f"/clean/{rid}"} for rid,sc in res]
    return jsonify({"topk": out})

@app.get("/clean/<path:fname>")
def serve_clean(fname): return send_from_directory(CLEAN_DIR, fname)

if __name__=="__main__":
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        cam.release()
