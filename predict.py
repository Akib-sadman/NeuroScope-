import argparse
import os
import csv
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from dataset import build_transforms, CLASS_NAMES, collect_samples
from model import SwinBackbone, MRIClassifier


def load_model(checkpoint_path, device):
    encoder = SwinBackbone(pretrained=False)
    model = MRIClassifier(encoder, num_classes=len(CLASS_NAMES))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, img_path, device, transform):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred_idx = int(probs.argmax())
    return CLASS_NAMES[pred_idx], probs.tolist()


def run_single(args, model, device, transform):
    label, probs = predict_image(model, args.image, device, transform)
    print(f"\nImage: {args.image}")
    print(f"Predicted class: {label}\n")
    for c, p in zip(CLASS_NAMES, probs):
        print(f"  {c:<12} {p * 100:6.2f}%")


def run_folder(args, model, device, transform):
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths.extend(Path(args.image_dir).glob(ext))
    if not paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    rows = []
    for p in paths:
        label, probs = predict_image(model, str(p), device, transform)
        conf = max(probs) * 100
        rows.append([p.name, label, f"{conf:.2f}%"])
        print(f"{p.name:<40} -> {label:<12} ({conf:.2f}%)")

    out_csv = os.path.join(args.out_dir, "predictions.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "confidence"])
        writer.writerows(rows)
    print(f"\nSaved predictions to {out_csv}")


def run_labeled(args, model, device, transform):
    """Evaluate on a labeled folder with the same class-subfolder layout as training."""
    samples = collect_samples(args.data_dir)
    y_true, y_pred = [], []

    for path, label_idx in samples:
        pred_label, _ = predict_image(model, path, device, transform)
        y_true.append(label_idx)
        y_pred.append(CLASS_NAMES.index(pred_label))

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"\nAccuracy on {args.data_dir}: {acc:.4f}\n")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix (New Test Set)")
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "test_confusion_matrix.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/mri_classifier.pt",
                         help="Path to the trained mri_classifier.pt")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a single image to classify")
    parser.add_argument("--image_dir", type=str, default=None,
                         help="Folder of unlabeled images to classify in batch")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Labeled folder (Glioma/Meningioma/Pituitary/'No Tumor' subfolders) to evaluate")
    args = parser.parse_args()

    if not any([args.image, args.image_dir, args.data_dir]):
        parser.error("Provide one of --image, --image_dir, or --data_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = load_model(args.checkpoint, device)
    transform = build_transforms(train=False)

    if args.image:
        run_single(args, model, device, transform)
    elif args.image_dir:
        run_folder(args, model, device, transform)
    elif args.data_dir:
        run_labeled(args, model, device, transform)


if __name__ == "__main__":
    main()