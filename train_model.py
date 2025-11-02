import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import YGOCardDataset
from model_setup import create_model  # from your model_setup.py

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "yugioh_images_processed"
BATCH_SIZE = 16
NUM_EPOCHS_PHASE1 = 10   # 🔹 First phase (frozen backbone)
NUM_EPOCHS_PHASE2 = 10   # 🔹 Second phase (fine-tune backbone)
LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_model.pth"

# ----------------------------
# 1. BUILD GLOBAL LABEL MAPPING
# ----------------------------
train_csv = os.path.join(DATA_DIR, "train", "train_labels.csv")
with open(train_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    labels = sorted(set(row["label"] for row in reader))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

num_classes = len(label_to_idx)
print(f"✅ Found {num_classes} unique labels")

# ----------------------------
# 2. DEFINE TRANSFORMS (Data Augmentation)
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# 3. LOAD DATASETS
# ----------------------------
train_dataset = YGOCardDataset(
    csv_file=os.path.join(DATA_DIR, "train", "train_labels.csv"),
    img_dir=os.path.join(DATA_DIR, "train"),
    transform=train_transform,
    label_to_idx=label_to_idx
)

val_dataset = YGOCardDataset(
    csv_file=os.path.join(DATA_DIR, "val", "val_labels.csv"),
    img_dir=os.path.join(DATA_DIR, "val"),
    transform=test_transform,
    label_to_idx=label_to_idx
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------
# 4. MODEL SETUP
# ----------------------------
def train_phase(model, optimizer, scheduler, num_epochs, phase_name):
    global best_val_acc
    for epoch in range(num_epochs):
        print(f"\n[{phase_name}] Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # --- Training ---
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_corrects.double() / total

        # --- Validation ---
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_corrects.double() / val_total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved new best model with acc: {val_acc:.4f}")

# ----------------------------
# 5. TRAINING PHASE 1 (frozen backbone)
# ----------------------------
print("\n🔥 PHASE 1: Training classifier head (frozen backbone)")
model = create_model(num_classes, freeze_backbone=True).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR_PHASE1, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
best_val_acc = 0.0

train_phase(model, optimizer, scheduler, NUM_EPOCHS_PHASE1, "Phase 1")

# ----------------------------
# 6. TRAINING PHASE 2 (unfreeze entire model)
# ----------------------------
print("\n🔓 PHASE 2: Fine-tuning full model (unfrozen backbone)")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LR_PHASE2, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

train_phase(model, optimizer, scheduler, NUM_EPOCHS_PHASE2, "Phase 2")

print("\n✅ Training complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")


