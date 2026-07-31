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
# Set to False once best_model.pth is copied into this project folder
USE_STUB: bool = True

# Path to your trained model checkpoint (best_model.pth from Crack/checkpoints/)
# Copy the file here and update the path:
REAL_MODEL_PATH: str = "best_model.pth"

# ── Model architecture settings (must match what was used during training) ─────
_ENCODER_NAME    = "resnet101"   # from config.py — upgraded from resnet50
_IMAGE_SIZE      = 512           # from config.py
_IMG_MEAN        = (0.485, 0.456, 0.406)   # ImageNet normalisation
_IMG_STD         = (0.229, 0.224, 0.225)
_CRACK_THRESHOLD = 0.5           # from config.py — pixels above this = crack

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
    Load the trained DeepLabV3+ (ResNet-101) crack segmentation model.
    Matches the exact pattern used in Crack/src/predict.py :: get_model()
    """
    import torch
    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.DeepLabV3Plus(
        encoder_name    = _ENCODER_NAME,   # resnet101
        encoder_weights = None,            # no pretrained weights — loading our checkpoint
        in_channels     = 3,
        classes         = 1,
        activation      = None,            # raw logits, sigmoid applied in inference
    )

    ckpt = torch.load(REAL_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return (model, device)


def _run_real_model(model_bundle, image: Image.Image) -> dict:
    """
    Run DeepLabV3+ inference on a PIL Image.
    Matches the exact logic in Crack/src/predict.py :: predict_crack()
    """
    import torch
    import cv2
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    model, device = model_bundle

    # ── Preprocessing (same as predict.py) ────────────────────────────────────
    transform = A.Compose([
        A.Resize(_IMAGE_SIZE, _IMAGE_SIZE),
        A.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
        ToTensorV2(),
    ])

    raw_rgb = np.array(image.convert("RGB"))
    orig_h, orig_w = raw_rgb.shape[:2]
    img_tensor = transform(image=raw_rgb)["image"].unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().float().cpu().numpy()

    # ── Resize back to original ────────────────────────────────────────────────
    conf_map = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    bin_mask  = (conf_map > _CRACK_THRESHOLD).astype(np.uint8)

    # ── Stats ─────────────────────────────────────────────────────────────────
    crack_pixels  = int(bin_mask.sum())
    total_pixels  = bin_mask.size
    area_fraction = round(crack_pixels / total_pixels * 100, 3)
    crack_detected = crack_pixels > 0
    confidence     = float(conf_map[bin_mask.astype(bool)].mean()) if crack_detected else 0.0
    severity       = _area_to_severity(area_fraction) if crack_detected else "None"

    # ── Infer crack type from area fraction ───────────────────────────────────
    if not crack_detected:
        crack_type = "No Crack"
    elif area_fraction < 1.0:
        crack_type = "Hairline"
    elif area_fraction < 4.0:
        crack_type = "Shrinkage"
    elif area_fraction < 8.0:
        crack_type = "Structural"
    else:
        crack_type = "Settlement"

    # ── Bounding boxes from connected components ───────────────────────────────
    boxes = []
    num_instances = 0
    if crack_detected:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 20:
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
        num_instances = len(boxes)

    # ── Estimated width (proxy from area / diagonal) ───────────────────────────
    width_mm = 0.0
    if crack_detected:
        diag_px  = (orig_w ** 2 + orig_h ** 2) ** 0.5
        width_px = (area_fraction / 100) * diag_px * 0.08
        width_mm = round(min(width_px * 0.264583, 2.0), 2)

    # ── Annotated overlay ─────────────────────────────────────────────────────
    annotated = _overlay_mask_real(image, bin_mask, boxes, severity, crack_type, confidence)

    return {
        "crack_detected"     : crack_detected,
        "confidence"         : round(confidence, 3),
        "crack_type"         : crack_type,
        "severity_estimate"  : severity,
        "area_fraction"      : area_fraction,
        "num_instances"      : num_instances,
        "estimated_width_mm" : width_mm,
        "bounding_boxes"     : boxes,
        "annotated_image"    : annotated,
        "model_mode"         : "real",
    }


def _overlay_mask_real(image: Image.Image, bin_mask,
                       boxes: list, severity: str,
                       crack_type: str, confidence: float) -> Image.Image:
    """Red semi-transparent crack overlay + bounding boxes. No watermark."""
    import numpy as np
    import cv2

    colors = {
        "Minor"   : (34,  197,  94),
        "Moderate": (245, 158,  11),
        "Severe"  : (239,  68,  68),
        "Critical": (124,  58, 237),
        "None"    : (100, 100, 100),
    }
    bgr_color = colors.get(severity, (239, 68, 68))[::-1]

    raw_rgb = np.array(image.convert("RGB"))
    overlay  = raw_rgb.copy()
    overlay[bin_mask.astype(bool)] = [255, 0, 0]
    blended  = (0.6 * raw_rgb + 0.4 * overlay).astype(np.uint8)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(blended, (x1, y1), (x2, y2), bgr_color, 2)
        label = f"{crack_type} {confidence*100:.0f}%"
        cv2.rectangle(blended, (x1, y1 - 18), (x1 + len(label) * 7, y1), bgr_color, -1)
        cv2.putText(blended, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    h, w = blended.shape[:2]
    cv2.rectangle(blended, (0, h - 28), (230, h), (15, 23, 42), -1)
    cv2.putText(blended, "Live Model — DeepLabV3+", (4, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 1)

    return Image.fromarray(blended)



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
