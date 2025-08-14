# Flask backend: /predict + /health + /img-proxy (streaming) with lazy TF/YOLO.
# Research/education only — not medical advice.

import os
import time
import uuid
import traceback
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# Optional imports — loaded lazily
try:
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:
    load_model = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

# ----------------------------------------------------------------------
STATIC_DIR = "static"
MODELS_DIR = "models"
STATIC_TTL_SECONDS = 60 * 60  # auto-delete files older than 1 hour (adjust)

def ensure_static_dir():
    os.makedirs(STATIC_DIR, exist_ok=True)

def cleanup_static(ttl_seconds: int = STATIC_TTL_SECONDS, max_delete: int = 200):
    """Delete old files from static/ to prevent disk growth."""
    ensure_static_dir()
    now = time.time()
    deleted = 0
    try:
        for name in os.listdir(STATIC_DIR):
            path = os.path.join(STATIC_DIR, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > ttl_seconds:
                try:
                    os.remove(path)
                    deleted += 1
                    if deleted >= max_delete:
                        break
                except Exception:
                    pass
    except FileNotFoundError:
        ensure_static_dir()

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
# TODO: Restrict CORS in production
CORS(app)

# ----------------------------- Config ---------------------------------
ensure_static_dir()
os.makedirs(MODELS_DIR, exist_ok=True)

_FACE_MODEL_PATH = os.path.join(MODELS_DIR, "pcos_detector_158.h5")
_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "bestv8.pt")
_ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

_FACE_SIZE = (100, 100)
# Must match the training output order
_FACE_LABELS = ["non_pcos", "unhealthy"]
# ----------------------------------------------------------------------


# ------------------------- Lazy model cache ---------------------------
_face_model = None
_yolo_model = None

def _lazy_face():
    """Load face classifier once (lazily)."""
    global _face_model
    if _face_model is None:
        if load_model is None:
            raise RuntimeError("TensorFlow/Keras is not installed.")
        if not os.path.exists(_FACE_MODEL_PATH):
            raise FileNotFoundError(f"Face model missing at '{_FACE_MODEL_PATH}'")
        _face_model = load_model(_FACE_MODEL_PATH)
    return _face_model

def _lazy_yolo():
    """Load YOLO model once (lazily)."""
    global _yolo_model
    if _yolo_model is None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed.")
        if not os.path.exists(_YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model missing at '{_YOLO_MODEL_PATH}'")
        _yolo_model = YOLO(_YOLO_MODEL_PATH)
    return _yolo_model
# ----------------------------------------------------------------------


# ----------------------------- IO utils -------------------------------
def _save_upload(field: str) -> Tuple[str, str]:
    """
    Save uploaded file to static/ with a UUID filename.
    IMPORTANT: We assume the client already sent upright pixels.
    Converts to RGB before saving for consistency.
    Returns (fs_path, web_path). If not present, returns ("","").
    """
    f = request.files.get(field)
    if not f:
        return "", ""
    if f.mimetype not in _ALLOWED_MIME:
        raise ValueError(f"Unsupported file type for {field}. Allowed: jpeg/png/webp")

    ensure_static_dir()

    ext = os.path.splitext(secure_filename(f.filename or ""))[1].lower() or ".jpg"
    uid = f"{uuid.uuid4().hex}{ext}"
    fs_path = os.path.join(STATIC_DIR, uid)

    try:
        img = Image.open(f.stream).convert("RGB")
        save_kwargs: Dict[str, Any] = {}
        fmt = None
        if ext in [".jpg", ".jpeg"]:
            fmt = "JPEG"
            save_kwargs["quality"] = 92
            save_kwargs["optimize"] = True
        elif ext == ".png":
            fmt = "PNG"
        elif ext == ".webp":
            fmt = "WEBP"
            save_kwargs["quality"] = 92
        img.save(fs_path, format=fmt, **save_kwargs)
    except Exception:
        # Fallback: stream raw bytes if PIL fails
        try:
            f.stream.seek(0)
        except Exception:
            pass
        f.save(fs_path)

    return fs_path, f"/static/{uid}"
# ----------------------------------------------------------------------


# --------------------------- Inference --------------------------------
def _predict_face(fs_path: str) -> Tuple[str, List[float]]:
    """Run face classifier and return (label, probability_list)."""
    img = Image.open(fs_path).convert("RGB").resize(_FACE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    probs = _lazy_face().predict(arr, verbose=0)[0]
    probs_list = [float(x) for x in probs]
    label = _FACE_LABELS[int(np.argmax(probs))]
    return label, probs_list

def _predict_yolo(fs_path: str) -> Tuple[str, str, List[str]]:
    """
    Run YOLO detection on X-ray.
    Returns (xray_label_text, visualization_web_path, found_class_names).
    """
    model = _lazy_yolo()
    res = model.predict(source=fs_path, verbose=False)[0]

    # Save visualization image
    vis_web = ""
    try:
        ensure_static_dir()
        vis_name = f"yolo_vis_{uuid.uuid4().hex}.jpg"
        vis_path = os.path.join(STATIC_DIR, vis_name)
        vis = res.plot()  # BGR array
        try:
            import cv2  # type: ignore
            import numpy as _np  # noqa
            cv2.imwrite(vis_path, vis)
        except Exception:
            Image.fromarray(vis[..., ::-1]).save(vis_path)
        vis_web = f"/static/{vis_name}"
    except Exception:
        pass

    # Extract class names found
    found: List[str] = []
    try:
        names = res.names if hasattr(res, "names") else getattr(model, "names", {})
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            for cls_id in res.boxes.cls.cpu().numpy().astype(int).tolist():
                nm = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                found.append(str(nm))
    except Exception:
        found = []

    xray_label = "PCOS symptoms detected in X-ray" if found else "No PCOS symptoms detected in X-ray"
    return xray_label, vis_web, found
# ----------------------------------------------------------------------


# ------------------------ Risk normalization --------------------------
_POS_FACE_LABELS = {"unhealthy", "pcos", "positive", "pcos_positive"}

def _face_is_positive(label: Optional[str]) -> bool:
    t = (label or "").strip().lower()
    return t in _POS_FACE_LABELS

def _xray_is_positive(found_labels: Optional[List[str]], xray_label: Optional[str]) -> bool:
    if isinstance(found_labels, list):
        return len(found_labels) > 0
    t = (xray_label or "").lower()
    return ("pcos" in t and "no " not in t and "non " not in t)

def _combine(face_pos: bool, xray_pos: bool) -> Tuple[str, str]:
    if face_pos and xray_pos:
        return "high", "High risk: Both modalities indicate PCOS symptoms."
    if face_pos or xray_pos:
        return "moderate", "Moderate risk: One modality suggests PCOS symptoms."
    return "low", "Low risk: No PCOS detected by either modality."
# ----------------------------------------------------------------------


# ----------------------------- Routes ---------------------------------
@app.get("/health")
def health():
    try:
        ok_face = os.path.exists(_FACE_MODEL_PATH)
        ok_yolo = os.path.exists(_YOLO_MODEL_PATH)
        return jsonify({"status": "ok", "models": {"face": ok_face, "yolo": ok_yolo}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# --------- Streaming Image Proxy (no saving, fixes CORS) ---------------
from urllib.parse import urlparse
import requests

ALLOWED_IMG_HOSTS = {
    "as2.ftcdn.net",
    "static.wixstatic.com",
    "resources.ama.uk.com",
    "www.emjreviews.com",
}

@app.get("/img-proxy")
def img_proxy():
    src = request.args.get("url", "")
    if not src:
        return jsonify({"error": "missing url"}), 400

    host = urlparse(src).netloc.lower()
    if host not in ALLOWED_IMG_HOSTS:
        return jsonify({"error": f"host '{host}' not allowed"}), 403

    try:
        r = requests.get(
            src,
            timeout=10,
            headers={
                # Some CDNs require a UA/Accept
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

    resp = Response(r.content, status=r.status_code)
    resp.headers["Content-Type"] = r.headers.get("Content-Type", "image/jpeg")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    # allow the front-end to read the blob
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp
# ----------------------------------------------------------------------


@app.post("/predict")
def predict():
    # Trim old files on every request (fast, bounded)
    cleanup_static()

    try:
        # Read exactly the fields your frontend sends (face_img / xray_img)
        face_path, face_web = _save_upload("face_img")
        xray_path, xray_web = _save_upload("xray_img")
        if not face_path and not xray_path:
            return jsonify({"message": "Provide at least one image: face_img or xray_img"}), 400

        out: Dict[str, Any] = {"ok": True}

        # ----- Face -----
        face_label: Optional[str] = None
        if face_path:
            lbl, probs = _predict_face(face_path)
            out["face_pred"] = lbl
            out["face_scores"] = probs
            out["face_img"] = face_web
            face_label = lbl

        # ----- X-ray (YOLO) -----
        xray_label: Optional[str] = None
        found_labels: Optional[List[str]] = None
        if xray_path:
            xlbl, vis_web, found = _predict_yolo(xray_path)
            out["xray_pred"] = xlbl
            out["yolo_vis"] = vis_web
            out["found_labels"] = found
            out["xray_img"] = xray_web
            xray_label = xlbl
            found_labels = found

        # ----- Risk mapping -----
        face_pos = _face_is_positive(face_label)
        xray_pos = _xray_is_positive(found_labels, xray_label)

        out["face_risk"] = ("high" if face_pos else "low") if face_label else "unknown"
        out["xray_risk"] = ("high" if xray_pos else "low") if xray_label else "unknown"

        overall_risk, combined_sentence = _combine(face_pos, xray_pos)
        out["overall_risk"] = overall_risk
        out["combined"] = combined_sentence

        return jsonify(out)

    except ValueError as ve:
        return jsonify({"message": str(ve)}), 400
    except FileNotFoundError as fe:
        return jsonify({"message": str(fe)}), 500
    except Exception:
        traceback.print_exc()
        return jsonify({"message": "Internal server error"}), 500


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dev: run backend on 5000, Vite proxy will forward /predict, /health, /img-proxy
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
