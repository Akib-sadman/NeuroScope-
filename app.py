import base64
import io
import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image

from dataset import build_transforms, CLASS_NAMES
from model import SwinBackbone, MRIClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(APP_DIR, "outputs", "mri_classifier.pt")

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = build_transforms(train=False)
_model = None


def get_model():
    """Lazily load the trained classifier once and reuse it across requests."""
    global _model
    if _model is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found at {CHECKPOINT_PATH}. Run train.py first."
            )
        encoder = SwinBackbone(pretrained=False)
        model = MRIClassifier(encoder, num_classes=len(CLASS_NAMES))
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _model = model
    return _model


def compute_gradcam(model, x):
    """
    Grad-CAM-style relevance map. Prefers an earlier Swin stage (finer
    spatial grid, e.g. 14x14) for a sharper, less blocky heatmap than the
    final 7x7 stage; falls back to the final-stage tokens if that stage
    isn't reachable for any reason (different timm version, etc).
    """
    activation, gradient = {}, {}
    target_layer = None
    try:
        target_layer = model.encoder.backbone.layers[2]
    except (AttributeError, IndexError):
        target_layer = None

    if target_layer is not None:
        def fwd_hook(_module, _inp, out):
            if out.dim() == 4:
                b, h, w, c = out.shape
                out = out.reshape(b, h * w, c)
            activation["value"] = out
            out.register_hook(lambda g: gradient.__setitem__("value", g))

        handle = target_layer.register_forward_hook(fwd_hook)
        try:
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            class_idx = int(probs.argmax().item())
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
        finally:
            handle.remove()

        if "value" in activation and "value" in gradient:
            act = activation["value"][0]           # (N, C)
            grad = gradient["value"][0]             # (N, C)
            alpha = grad.mean(dim=0)                # (C,) — classic Grad-CAM channel weights
            relevance = F.relu((act * alpha.unsqueeze(0)).sum(dim=-1))  # (N,)
            side = int(round(relevance.shape[0] ** 0.5))
            heatmap = relevance.reshape(side, side)
            heatmap = (heatmap / (heatmap.max() + 1e-8)).detach().cpu().numpy()
            return heatmap, class_idx, probs

    # Fallback: final-stage 7x7 token grid.
    tokens = model.encoder.forward_tokens(x)
    tokens.retain_grad()
    pooled = tokens.mean(dim=1)
    logits = model.head(pooled)
    probs = F.softmax(logits, dim=1)[0]
    class_idx = int(probs.argmax().item())
    model.zero_grad(set_to_none=True)
    logits[0, class_idx].backward()
    grad = tokens.grad[0]
    act = tokens[0].detach()
    alpha = grad.mean(dim=0)
    relevance = F.relu((act * alpha.unsqueeze(0)).sum(dim=-1))
    side = int(round(relevance.shape[0] ** 0.5))
    heatmap = relevance.reshape(side, side)
    heatmap = (heatmap / (heatmap.max() + 1e-8)).detach().cpu().numpy()
    return heatmap, class_idx, probs


def make_overlay(pil_img, heatmap, size=(224, 224), max_alpha=0.65, gamma=1.8):
    """
    Blend a normalized (H, W) heatmap onto the original image with
    per-pixel alpha proportional to relevance (steepened by `gamma`), so
    only genuinely high-relevance regions are tinted and the rest of the
    image stays close to the original grayscale.
    """
    base = pil_img.convert("RGB").resize(size)
    hm_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
    hm_arr = np.clip(np.array(hm_img) / 255.0, 0.0, 1.0)

    colored = (cm.get_cmap("inferno")(hm_arr)[:, :, :3] * 255).astype(np.float32)
    base_arr = np.array(base).astype(np.float32)

    alpha_map = (hm_arr ** gamma) * max_alpha
    alpha_map = alpha_map[:, :, None]

    blended_arr = base_arr * (1 - alpha_map) + colored * alpha_map
    blended = Image.fromarray(blended_arr.astype(np.uint8))

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def looks_like_grayscale_scan(pil_img, sample_size=64, max_mean_channel_diff=12.0):
    """
    Cheap heuristic pre-filter: MRI scans are essentially grayscale
    (R ~ G ~ B per pixel). Ordinary photos have much higher per-pixel
    channel variance. Not foolproof (a grayscale photo of anything else
    still passes this), but it catches the obvious cases cheaply, before
    running the model at all.
    """
    small = pil_img.convert("RGB").resize((sample_size, sample_size))
    arr = np.asarray(small).astype(np.float32)
    channel_range = arr.max(axis=2) - arr.min(axis=2)  # per-pixel max-min across R,G,B
    mean_diff = float(channel_range.mean())
    return mean_diff <= max_mean_channel_diff, mean_diff


def prediction_confidence_ok(probs, min_confidence=0.55, max_entropy=1.15):
    """
    Second-layer guard, based on the model's own output: even a grayscale
    image that isn't a brain MRI (a different body part, a random
    grayscale photo) can still get a confident-looking prediction from a
    closed 4-class softmax. Low max-probability or high entropy across the
    4 classes is a signal the model doesn't actually recognize the input.
    max_entropy for 4 uniform classes is ln(4) ~= 1.386; 1.15 is a
    moderately strict cutoff below that ceiling.
    """
    probs_np = probs.detach().cpu().numpy()
    max_p = float(probs_np.max())
    entropy = float(-(probs_np * np.log(probs_np + 1e-8)).sum())
    return (max_p >= min_confidence and entropy <= max_entropy), max_p, entropy


CLASS_INFO = {
    "Glioma": "Gliomas arise from glial cells \u2014 the supportive tissue surrounding neurons in the "
              "brain. They range from slow-growing to fast-growing, and are the most common type of "
              "primary brain tumor.",
    "Meningioma": "Meningiomas arise from the meninges, the layers of tissue covering the brain and "
                  "spinal cord, rather than from brain tissue itself. Most are slow-growing and "
                  "non-cancerous.",
    "Pituitary": "Pituitary tumors develop in the pituitary gland, a small gland at the base of the "
                 "brain that regulates hormone production. Most are benign (pituitary adenomas).",
    "No Tumor": "No tumor was detected in this scan by the model.",
}


def describe_confidence(confidence, out_of_distribution):
    pct = confidence * 100
    if out_of_distribution:
        return (f"The model's confidence was {pct:.1f}%, but this result was flagged as uncertain "
                 "\u2014 treat it cautiously, it may not even be a recognizable brain MRI.")
    if pct >= 95:
        return f"The model was very confident here \u2014 {pct:.1f}%."
    if pct >= 80:
        return f"The model was fairly confident \u2014 {pct:.1f}%."
    return f"The model's confidence was moderate \u2014 {pct:.1f}%. Borderline cases like this are worth double-checking."


def describe_location(lateral_frac, front_frac, predicted_class):
    if predicted_class == "No Tumor":
        return "No region was highlighted since no tumor was detected."
    if predicted_class == "Pituitary":
        return ("Pituitary tumors form at a consistent location \u2014 the base of the brain \u2014 so "
                "the 3D view always highlights that spot rather than following this image.")
    side = "the right side" if (lateral_frac or 0) > 0.15 else \
           "the left side" if (lateral_frac or 0) < -0.15 else "roughly the midline"
    depth = "toward the front" if (front_frac or 0) > 0.15 else \
            "toward the back" if (front_frac or 0) < -0.15 else "centrally"
    return (f"Based on where the heatmap concentrated in this image, the highlighted patch in the "
            f"3D view is placed on {side}, {depth} of the brain.")


def build_explanation(ctx):
    cls = ctx.get("predicted_class", "Unknown")
    conf = ctx.get("confidence", 0.0)
    ood = ctx.get("out_of_distribution", False)
    lf = ctx.get("lateral_frac")
    ff = ctx.get("front_frac")

    parts = [f"This scan was classified as {cls}."]
    parts.append(describe_confidence(conf, ood))
    if cls in CLASS_INFO:
        parts.append(CLASS_INFO[cls])
    parts.append(describe_location(lf, ff, cls))
    parts.append("Reminder: this is a research demo, not a diagnostic tool \u2014 always consult a "
                 "radiologist or physician for an actual medical reading.")
    return " ".join(parts)


INTENT_KEYWORDS = {
    "confidence": ["confiden", "sure", "certain", "accura"],
    "heatmap": ["heatmap", "heat map", "highlight", "yellow", "orange", "gradcam", "grad-cam"],
    "3d": ["3d", "brain model", "marker", "region", "where"],
    "class_info": ["what is", "tell me about", "meaning", "means"],
    "diagnosis": ["diagnos", "real", "trust", "accurate", "doctor", "worry"],
}


def match_intent(message):
    msg = message.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in msg for k in keywords):
            return intent
    return None


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    intent = data.get("intent") or match_intent(message)
    ctx = data.get("context") or {}
    cls = ctx.get("predicted_class")

    if not cls:
        return jsonify({"reply": "Upload and classify a scan first, then ask me about the result."})

    if intent == "confidence":
        reply = describe_confidence(ctx.get("confidence", 0.0), ctx.get("out_of_distribution", False))
    elif intent == "heatmap":
        reply = ("The heatmap overlay on the scan shows which parts of THIS image most influenced "
                 "the model's decision \u2014 brighter yellow/orange means higher influence. It's "
                 "derived directly from the model's gradients on your uploaded image.")
    elif intent == "3d":
        reply = describe_location(ctx.get("lateral_frac"), ctx.get("front_frac"), cls) + (
            " Keep in mind the 3D brain is schematic \u2014 it shows a typical/associated region for "
            "this tumor type, not a verified 3D location of an actual tumor."
        )
    elif intent == "class_info":
        reply = CLASS_INFO.get(cls, "I don't have information on that class.")
    elif intent == "diagnosis":
        reply = ("No \u2014 this is a research/educational demo, not a diagnostic device. Predictions "
                 "can be wrong, especially on images unlike the training data. Always consult a "
                 "qualified radiologist or physician for an actual diagnosis.")
    elif intent == "explain_result" or not message:
        reply = build_explanation(ctx)
    else:
        reply = ("I can explain the classification, confidence, heatmap, or 3D reference for this "
                 "scan \u2014 try asking about one of those, or use the quick buttons above.")

    return jsonify({"reply": reply})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    img = Image.open(request.files["image"].stream).convert("RGB")

    # --- OOD guard 1: does this even look like a grayscale scan? ---
    is_grayscale, channel_diff = looks_like_grayscale_scan(img)
    if not is_grayscale:
        return jsonify({
            "out_of_distribution": True,
            "ood_reason": "not_grayscale",
            "ood_message": "This doesn't look like a grayscale MRI scan (too much color "
                            "variation). This classifier only recognizes brain MRI slices.",
        })

    x = _transform(img).unsqueeze(0).to(device)
    heatmap, class_idx, probs = compute_gradcam(model, x)

    # --- OOD guard 2: is the model itself confident? ---
    conf_ok, max_p, entropy = prediction_confidence_ok(probs)

    side_n = heatmap.shape[0]
    col_weights = heatmap.sum(axis=0)
    row_weights = heatmap.sum(axis=1)
    total_weight = float(heatmap.sum())
    if total_weight > 1e-6:
        centroid_x = float((col_weights * np.arange(side_n)).sum() / total_weight) / max(side_n - 1, 1)
        centroid_y = float((row_weights * np.arange(side_n)).sum() / total_weight) / max(side_n - 1, 1)
    else:
        centroid_x, centroid_y = 0.5, 0.5

    # Continuous position, used for a focused 3D patch instead of a whole
    # hemisphere wash. Assumes an axial-style slice (image-top ~= front);
    # a simplification that won't hold for every scan orientation.
    lateral_frac = (centroid_x - 0.5) * 2.0   # -1 (left) .. +1 (right)
    front_frac = (0.5 - centroid_y) * 2.0     # -1 (back) .. +1 (front)

    if centroid_x < 0.42:
        lateral_bias = "left"
    elif centroid_x > 0.58:
        lateral_bias = "right"
    else:
        lateral_bias = "center"

    overlay = make_overlay(img, heatmap)

    return jsonify({
        "predicted_class": CLASS_NAMES[class_idx],
        "confidence": float(probs[class_idx].item()),
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs.tolist())},
        "heatmap_overlay": overlay,
        "lateral_bias": lateral_bias,
        "lateral_frac": lateral_frac,
        "front_frac": front_frac,
        "out_of_distribution": not conf_ok,
        "ood_reason": None if conf_ok else "low_confidence",
        "ood_message": None if conf_ok else (
            "The model isn't confident this is a recognizable brain MRI slice "
            "(low confidence / high uncertainty across classes). Treat this "
            "prediction with caution — it may not be a brain MRI at all."
        ),
    })


if __name__ == "__main__":
    # use_reloader=False: the auto-reloader restarts the whole process on
    # every file save, which on Windows can crash mid-import if it fires
    # while a save is still in flight. Since this app loads a large model
    # once at startup, restarting isn't something we want happening
    # automatically anyway — just stop and re-run `python app.py` after
    # editing.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)