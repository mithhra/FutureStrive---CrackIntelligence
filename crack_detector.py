"""
crack_detector.py
------------------
Reactive Crack Detection — Model Wrapper

This file is the ONLY place you need to touch when plugging in your
real segmentation / detection model.  Everything else (AI assistant,
intent router, LLM report) reads the structured dict this file returns.

Output contract — run_crack_detection() always returns:
{
    "crack_detected"    : bool,
    "confidence"        : float,   # 0.0 – 1.0
    "crack_type"        : str,     # "Hairline" | "Structural" | "Shrinkage" | "Settlement" | "No Crack"
    "severity_estimate" : str,     # "Minor" | "Moderate" | "Severe" | "Critical" | "None"
    "area_fraction"     : float,   # % of image pixels classified as crack
    "num_instances"     : int,     # number of crack segments detected
    "estimated_width_mm": float,   # estimated real-world crack width in mm (approx)
    "bounding_boxes"    : list,    # list of [x1, y1, x2, y2] in pixel coords
    "annotated_image"   : PIL.Image or None,  # image with overlaid mask / boxes
    "model_mode"        : str,     # "real" | "stub"  — tells the UI which mode is active
}

HOW TO PLUG IN YOUR REAL MODEL
-------------------------------
1.  Set REAL_MODEL_PATH below to the path of your saved model file
    (.pt for PyTorch / .h5 for Keras / .onnx for ONNX etc.)
2.  Implement _run_real_model() with your actual inference code
3.  Set USE_STUB = False

The stub mode runs with NO model installed and produces realistic
synthetic outputs so the full UI + LLM pipeline can be developed
and tested immediately.
"""

import io
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Configuration ─────────────────────────────────────────────────────────────
# Set to False once your real model is ready
USE_STUB: bool = True

# Path to your saved model file — update this when your model is ready
REAL_MODEL_PATH: str = "crack_segmentation_model.pt"   # e.g. .pt / .h5 / .onnx

# ── Severity thresholds (area_fraction %) ─────────────────────────────────────
# Used by the stub and also by any real model that returns raw area fraction
SEVERITY_THRESHOLDS = {
    "Minor"    : (0.0,  1.5),   # < 1.5% of image is crack
    "Moderate" : (1.5,  4.0),
    "Severe"   : (4.0,  8.0),
    "Critical" : (8.0, 100.0),
}

CRACK_TYPES = ["Hairline", "Structural", "Shrinkage", "Settlement"]

# ── IS code permissible crack widths (mm) by type ─────────────────────────────
IS_WIDTH_LIMITS = {
    "Hairline"   : 0.10,
    "Shrinkage"  : 0.20,
    "Structural" : 0.20,
    "Settlement" : 0.30,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STUB MODE  — synthetic realistic outputs (no model required)
# ═══════════════════════════════════════════════════════════════════════════════

def _stub_detection(image: Image.Image) -> dict:
    """
    Generates a realistic synthetic detection result for UI / LLM testing.
    Uses actual image pixel data to seed randomness so the same image
    always produces the same output.
    """
    # Seed from image content for repeatability
    arr = np.array(image.convert("L")).flatten()
    seed = int(arr[:100].sum()) % 10000
    rng = random.Random(seed)

    crack_detected = rng.random() > 0.20  # 80% chance of detecting a crack

    if not crack_detected:
        annotated = image.copy()
        _stamp_no_crack(annotated)
        return {
            "crack_detected"     : False,
            "confidence"         : round(rng.uniform(0.85, 0.99), 3),
            "crack_type"         : "No Crack",
            "severity_estimate"  : "None",
            "area_fraction"      : 0.0,
            "num_instances"      : 0,
            "estimated_width_mm" : 0.0,
            "bounding_boxes"     : [],
            "annotated_image"    : annotated,
            "model_mode"         : "stub",
        }

    # Crack detected — generate plausible features
    crack_type    = rng.choice(CRACK_TYPES)
    area_fraction = round(rng.uniform(0.5, 12.0), 2)
    severity      = _area_to_severity(area_fraction)
    confidence    = round(rng.uniform(0.72, 0.97), 3)
    num_instances = rng.randint(1, 5)
    width_mm      = round(rng.uniform(0.05, 0.8), 2)

    # Generate synthetic bounding boxes
    w, h = image.size
    boxes = []
    for _ in range(num_instances):
        x1 = rng.randint(0, w - 60)
        y1 = rng.randint(0, h - 30)
        x2 = min(w, x1 + rng.randint(40, 200))
        y2 = min(h, y1 + rng.randint(15, 80))
        boxes.append([x1, y1, x2, y2])

    annotated = _draw_stub_overlay(image.copy(), boxes, severity, crack_type, confidence)

    return {
        "crack_detected"     : True,
        "confidence"         : confidence,
        "crack_type"         : crack_type,
        "severity_estimate"  : severity,
        "area_fraction"      : area_fraction,
        "num_instances"      : num_instances,
        "estimated_width_mm" : width_mm,
        "bounding_boxes"     : boxes,
        "annotated_image"    : annotated,
        "model_mode"         : "stub",
    }


def _area_to_severity(area_fraction: float) -> str:
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= area_fraction < hi:
            return label
    return "Critical"


def _draw_stub_overlay(img: Image.Image, boxes: list,
                       severity: str, crack_type: str,
                       confidence: float) -> Image.Image:
    """Draw bounding boxes and labels onto the image."""
    colors = {
        "Minor"   : "#22C55E",
        "Moderate": "#F59E0B",
        "Severe"  : "#EF4444",
        "Critical": "#7C3AED",
    }
    color = colors.get(severity, "#EF4444")

    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{crack_type} [{severity}] {confidence*100:.0f}%"
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=color)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x1 + 2, y1 - 16), label, fill="white", font=font)

    # Watermark corner
    w, h = img.size
    draw.rectangle([0, h - 28, 220, h], fill="#0F172A")
    draw.text((4, h - 22), "⚠ DEMO MODE — Stub Output", fill="#F59E0B")
    return img


def _stamp_no_crack(img: Image.Image) -> None:
    """Stamp 'No Crack Detected' overlay."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle([0, h - 28, 240, h], fill="#0F172A")
    draw.text((4, h - 22), "✓ No Crack Detected (Demo Mode)", fill="#22C55E")


# ═══════════════════════════════════════════════════════════════════════════════
# REAL MODEL  — plug your model in here
# ═══════════════════════════════════════════════════════════════════════════════

def _load_real_model():
    """
    Load your trained segmentation model.
    Called once — cache the result yourself or use @st.cache_resource in app.py.

    EXAMPLE for a PyTorch model:
        import torch
        model = torch.load(REAL_MODEL_PATH, map_location="cpu")
        model.eval()
        return model

    EXAMPLE for an ONNX model:
        import onnxruntime as ort
        session = ort.InferenceSession(REAL_MODEL_PATH)
        return session

    EXAMPLE for a Keras / TF SavedModel:
        import tensorflow as tf
        model = tf.saved_model.load(REAL_MODEL_PATH)
        return model
    """
    raise NotImplementedError(
        "Real model not yet connected. "
        "Implement _load_real_model() and _run_real_model() in crack_detector.py, "
        "then set USE_STUB = False."
    )


def _run_real_model(model, image: Image.Image) -> dict:
    """
    Run your model on a PIL image.

    You must populate all keys in the output dict.
    See the docstring at the top of this file for the full contract.

    MINIMUM you need to fill in:
        crack_detected, confidence, crack_type, severity_estimate,
        area_fraction, num_instances, estimated_width_mm,
        bounding_boxes, annotated_image

    Set model_mode = "real" always.

    EXAMPLE skeleton:
        import torch, torchvision.transforms as T
        transform = T.Compose([T.Resize((512,512)), T.ToTensor()])
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
        mask = output["masks"][0].squeeze().numpy()  # H x W binary mask
        area_fraction = float(mask.mean() * 100)
        ...
        annotated = _overlay_mask_on_image(image, mask)
        return {
            "crack_detected"    : area_fraction > 0.1,
            "confidence"        : float(output["scores"][0]),
            "crack_type"        : "Structural",   # from your classifier head
            "severity_estimate" : _area_to_severity(area_fraction),
            "area_fraction"     : area_fraction,
            "num_instances"     : int(len(output["masks"])),
            "estimated_width_mm": ...,
            "bounding_boxes"    : output["boxes"].tolist(),
            "annotated_image"   : annotated,
            "model_mode"        : "real",
        }
    """
    raise NotImplementedError("Implement _run_real_model() in crack_detector.py.")


def _overlay_mask_on_image(image: Image.Image, mask: np.ndarray,
                            color=(255, 50, 50, 120)) -> Image.Image:
    """
    Helper: overlay a binary segmentation mask on the original image.
    mask: H x W numpy array with values 0/1 or 0/255
    """
    img_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mask_bin = (mask > 0.5).astype(np.uint8) * 255
    for y in range(mask_bin.shape[0]):
        for x in range(mask_bin.shape[1]):
            if mask_bin[y, x]:
                draw.point((x, y), fill=color)
    return Image.alpha_composite(img_rgba, overlay).convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  — this is what the app calls
# ═══════════════════════════════════════════════════════════════════════════════

def run_crack_detection(image: Image.Image) -> dict:
    """
    Main entry point.  Called from app.py with a PIL Image.

    Returns the standard detection dict (see module docstring).
    Falls back to stub mode if USE_STUB is True or if the real model
    raises any exception.
    """
    if USE_STUB or not os.path.exists(REAL_MODEL_PATH):
        return _stub_detection(image)

    try:
        model = _load_real_model()
        return _run_real_model(model, image)
    except NotImplementedError:
        return _stub_detection(image)
    except Exception as e:
        # Graceful fallback — never crash the UI
        result = _stub_detection(image)
        result["model_mode"] = f"stub_fallback:{str(e)[:60]}"
        return result


def detection_to_summary_text(det: dict) -> str:
    """
    Convert detection dict → plain-text summary for injection into LLM context.
    Called by llm_engine.py.
    """
    if not det.get("crack_detected"):
        return (
            "REACTIVE DETECTION RESULT: No crack detected.\n"
            f"Confidence: {det.get('confidence', 0)*100:.1f}%\n"
            f"Model Mode: {det.get('model_mode', 'stub')}\n"
        )

    limit = IS_WIDTH_LIMITS.get(det.get("crack_type", "Structural"), 0.20)
    width = det.get("estimated_width_mm", 0.0)
    exceeds = "YES — EXCEEDS IS 456:2000 LIMIT" if width > limit else "Within permissible limit"

    lines = [
        "=== REACTIVE CRACK DETECTION RESULT ===",
        f"Crack Detected       : YES",
        f"Detection Confidence : {det.get('confidence', 0)*100:.1f}%",
        f"Crack Type           : {det.get('crack_type', 'Unknown')}",
        f"Severity             : {det.get('severity_estimate', 'Unknown')}",
        f"Area Fraction        : {det.get('area_fraction', 0):.2f}% of image",
        f"Number of Instances  : {det.get('num_instances', 0)}",
        f"Est. Crack Width     : {width:.2f} mm  (IS 456 limit for this type: {limit} mm — {exceeds})",
        f"Model Mode           : {det.get('model_mode', 'stub')}",
    ]
    return "\n".join(lines)
