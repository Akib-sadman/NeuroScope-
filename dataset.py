import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMG_SIZE = 224  # Swin-Tiny default input resolution


def build_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


class BrainMRIDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_samples(data_dir):
    """
    Expects a folder layout like:
        data_dir/Glioma/*.jpg
        data_dir/Meningioma/*.jpg
        data_dir/Pituitary/*.jpg
        data_dir/No Tumor/*.jpg
    """
    data_dir = Path(data_dir)
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            alt = data_dir / cls.replace(" ", "_")
            cls_dir = alt if alt.exists() else cls_dir
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in cls_dir.glob(ext):
                samples.append((str(p), CLASS_TO_IDX[cls]))
    if not samples:
        raise RuntimeError(
            f"No images found under {data_dir}. Check that subfolders match {CLASS_NAMES}."
        )
    return samples


def split_dataset(data_dir, val_size=0.15, test_size=0.15, seed=42):
    samples = collect_samples(data_dir)
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(val_size + test_size), stratify=labels, random_state=seed
    )

    rel_test = test_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=rel_test, stratify=temp_labels, random_state=seed
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))
    return train_samples, val_samples, test_samples