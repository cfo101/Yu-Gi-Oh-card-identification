import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class YGOCardDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, label_to_idx=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        # Read CSV (filename, label)
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], row["label"]))

        # --- Label encoding ---
        if label_to_idx is None:
            # build new mapping if not provided
            labels = sorted(list(set(label for _, label in self.samples)))
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            # use shared mapping (recommended)
            self.label_to_idx = label_to_idx

        # reverse lookup (optional)
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Pre-compute numeric labels for efficiency
        self.samples = [
            (filename, self.label_to_idx[label])
            for filename, label in self.samples
            if label in self.label_to_idx
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label_idx = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_idx

def get_label_mapping(csv_file):
    labels = set()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.add(row["label"])
    labels = sorted(list(labels))
    return {label: idx for idx, label in enumerate(labels)}
