import torch
import torch.nn as nn
import pandas as pd
from torchvision import models

def create_model(num_classes, freeze_backbone=True):
    """
    Create a ResNet18 model for Yu-Gi-Oh card classification.
    """
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Optionally freeze earlier layers (speeds up training, reduces overfitting)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer with a new classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def define_loss_and_optimizer(model, lr=1e-4):
    """
    Define loss function and optimizer for training.
    """
    # CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer for adaptive gradient updates
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer

def count_num_classes(csv_path):
    df = pd.read_csv(csv_path)
    unique_classes = df['label'].nunique()
    print(f"Detected {unique_classes} unique card names.")
    return unique_classes

def get_model_setup(csv_path, lr=1e-4, freeze_backbone=True):
    num_classes = count_num_classes(csv_path)
    model = create_model(num_classes, freeze_backbone=freeze_backbone)
    criterion, optimizer = define_loss_and_optimizer(model, lr=lr)
    return model, criterion, optimizer


if __name__ == "__main__":
    # Example usage / quick test
    num_classes = count_num_classes("yugioh_images_processed/train/train_labels.csv")
    # Create model
    model = create_model(num_classes)

    # Define loss and optimizer
    criterion, optimizer = define_loss_and_optimizer(model)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"✅ Model ready on {device}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(model)

