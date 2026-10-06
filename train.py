import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from dataset import BrainMRIDataset, build_transforms, split_dataset, CLASS_NAMES
from model import JEPAModel, SwinBackbone, MRIClassifier


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pretrain_jepa(args, train_samples, device):
    ds = BrainMRIDataset(train_samples, transform=build_transforms(train=True))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.workers, pin_memory=True)

    model = JEPAModel(pretrained_backbone=True, ckpt_path=args.backbone_ckpt,
                       mask_ratio=args.mask_ratio, ema_decay=args.ema_decay).to(device)
    optim = torch.optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()) +
        list(model.decoder.parameters()) + [model.pos_embed],
        lr=args.lr, weight_decay=0.05,
    )

    history = {"epoch": [], "loss": []}
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"[JEPA pretrain] epoch {epoch}/{args.pretrain_epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            loss = model(imgs)
            optim.zero_grad()
            loss.backward()
            optim.step()
            model.update_target()
            running += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())
        epoch_loss = running / len(ds)
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        print(f"epoch {epoch}: JEPA loss = {epoch_loss:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.context_encoder.state_dict(), os.path.join(args.out_dir, "jepa_encoder.pt"))
    plot_pretrain_curve(history, args.out_dir)
    return model.context_encoder


def plot_pretrain_curve(history, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("JEPA loss (smooth L1)")
    plt.title("JEPA Pretraining Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "jepa_pretrain_loss.png"), dpi=150)
    plt.close()


def run_classifier(args, encoder, train_samples, val_samples, test_samples, device):
    train_ds = BrainMRIDataset(train_samples, transform=build_transforms(train=True))
    val_ds = BrainMRIDataset(val_samples, transform=build_transforms(train=False))
    test_ds = BrainMRIDataset(test_samples, transform=build_transforms(train=False))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = MRIClassifier(encoder, num_classes=len(CLASS_NAMES),
                           freeze_backbone=args.freeze_backbone).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_dl, desc=f"[Finetune] epoch {epoch}/{args.finetune_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            run_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = run_loss / total
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "mri_classifier.pt"))
    plot_training_curves(history, args.out_dir)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_dl, criterion, device, return_preds=True)
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")
    plot_confusion_matrix(y_true, y_pred, args.out_dir)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model


@torch.no_grad()
def evaluate(model, dl, criterion, device, return_preds=False):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    for imgs, labels in dl:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        run_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if return_preds:
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return run_loss / total, correct / total, y_true, y_pred


def plot_training_curves(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_acc"], marker="o", label="Train")
    axes[1].plot(history["epoch"], history["val_acc"], marker="o", label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset root with Glioma/Meningioma/Pituitary/'No Tumor' folders")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                         help="Optional path to a MedCLIP-Swin/MIMIC-EYE checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=15)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true",
                         help="Skip JEPA pretraining and fine-tune the ImageNet backbone directly")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_samples, val_samples, test_samples = split_dataset(args.data_dir)
    print(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    if args.skip_pretrain:
        encoder = SwinBackbone(pretrained=True, ckpt_path=args.backbone_ckpt).to(device)
    else:
        encoder = pretrain_jepa(args, train_samples, device)

    run_classifier(args, encoder, train_samples, val_samples, test_samples, device)


if __name__ == "__main__":
    main()