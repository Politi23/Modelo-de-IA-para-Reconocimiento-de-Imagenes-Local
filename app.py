import io
import base64
import logging
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    BlipProcessor, BlipForConditionalGeneration,
)
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

CLS_MODEL   = "microsoft/swin-large-patch4-window12-384"
DESC_MODEL  = "Salesforce/blip-image-captioning-large"
LABELS_CACHE = "labels_es.json"

def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except Exception:
            logger.warning("CUDA no compatible — usando CPU.")
    return "cpu"

DEVICE = get_device()

# Classification
cls_extractor = None
cls_model     = None
label_es      = {}

# Description
desc_processor = None
desc_model     = None

model_status = "loading"

TTA_TRANSFORMS = [
    T.Compose([]),
    T.Compose([T.RandomHorizontalFlip(p=1.0)]),
    T.Compose([T.RandomVerticalFlip(p=1.0)]),
    T.Compose([T.RandomRotation((90, 90))]),
    T.Compose([T.RandomRotation((-90, -90))]),
]


# ── Label translation ──────────────────────────────────────────────────────────
def translate_labels(labels: list[str]) -> dict[str, str]:
    import json, os
    if os.path.exists(LABELS_CACHE):
        logger.info("Cargando traducciones desde caché...")
        with open(LABELS_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Traduciendo 1000 labels (solo ocurre una vez)...")
    translator = GoogleTranslator(source="en", target="es")
    result = {}
    batch_size = 100
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i + batch_size]
        try:
            joined = "\n".join(batch)
            translated_text = translator.translate(joined)
            translated = translated_text.split("\n")
            if len(translated) != len(batch):
                translated = batch
            for en, es in zip(batch, translated):
                result[en] = es.strip() if es else en
        except Exception as e:
            logger.warning(f"Error en batch {i}: {e}")
            for en in batch:
                result[en] = en
        logger.info(f"  {min(i + batch_size, len(labels))}/{len(labels)} labels traducidos")

    with open(LABELS_CACHE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    global cls_extractor, cls_model, label_es
    global desc_processor, desc_model, model_status

    try:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        # 1. Classification (Swin-Large)
        logger.info(f"[1/2] Cargando clasificador ({CLS_MODEL})...")
        cls_extractor = AutoImageProcessor.from_pretrained(CLS_MODEL)
        cls_model = AutoModelForImageClassification.from_pretrained(
            CLS_MODEL, torch_dtype=dtype
        )
        cls_model.eval().to(DEVICE)

        raw_labels = [
            cls_model.config.id2label[i].split(",")[0].strip()
            for i in range(len(cls_model.config.id2label))
        ]
        label_es = translate_labels(raw_labels)

        # 2. Description (BLIP-large)
        logger.info(f"[2/2] Cargando descriptor ({DESC_MODEL})...")
        desc_processor = BlipProcessor.from_pretrained(DESC_MODEL)
        desc_model = BlipForConditionalGeneration.from_pretrained(
            DESC_MODEL, dtype=dtype
        )
        desc_model.eval().to(DEVICE)

        model_status = "ready"
        logger.info("Ambos modelos listos.")
    except Exception as e:
        model_status = "error"
        logger.error(f"Error cargando modelos: {e}")


# ── Inference helpers ──────────────────────────────────────────────────────────
def run_tta(image: Image.Image) -> torch.Tensor:
    all_probs = []
    for transform in TTA_TRANSFORMS:
        aug = transform(image)
        inputs = cls_extractor(images=aug, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        if DEVICE == "cuda":
            inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
        with torch.no_grad():
            logits = cls_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits.float(), dim=-1)[0]
        all_probs.append(probs)
    return torch.stack(all_probs).mean(dim=0)


def run_description(image: Image.Image) -> str:
    inputs = desc_processor(
        images=image,
        text="a detailed description of this image showing",
        return_tensors="pt"
    ).to(DEVICE)
    if DEVICE == "cuda":
        inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = desc_model.generate(
            **inputs,
            max_new_tokens=180,
            num_beams=5,
            repetition_penalty=1.3,
            length_penalty=1.2,
        )

    prompt_prefix = "a detailed description of this image showing"
    desc_en = desc_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    if desc_en.lower().startswith(prompt_prefix):
        desc_en = desc_en[len(prompt_prefix):].strip()

    try:
        desc_es = GoogleTranslator(source="en", target="es").translate(desc_en)
        return desc_es or desc_en
    except Exception:
        return desc_en


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "cls_model":  CLS_MODEL,
        "desc_model": DESC_MODEL,
        "device":     DEVICE,
        "gpu_name":   torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "tta":        len(TTA_TRANSFORMS),
        "status":     model_status,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model_status == "loading":
        return jsonify({"error": "El modelo aún se está cargando"}), 503
    if model_status == "error":
        return jsonify({"error": "Error al cargar el modelo"}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No se recibió imagen"}), 400

    try:
        image_data = data["image"].split(",")[1] if "," in data["image"] else data["image"]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Imagen inválida: {e}"}), 400

    # Classification
    probs = run_tta(image)
    top5  = torch.topk(probs, k=5)
    predictions = []
    for score, idx in zip(top5.values, top5.indices):
        en = cls_model.config.id2label[idx.item()].split(",")[0].strip()
        predictions.append({
            "label":      label_es.get(en, en),
            "confidence": round(score.item() * 100, 2),
        })

    # Description
    description = run_description(image)

    return jsonify({"description": description, "predictions": predictions})


if __name__ == "__main__":
    threading.Thread(target=load_model, daemon=True).start()
    app.run(debug=False, port=5000)
