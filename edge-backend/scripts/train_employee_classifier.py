#!/usr/bin/env python3
"""
Entrenamiento de clasificador employee/non_employee.

Estructura esperada:
dataset/
  train/
    employee/
    non_employee/
  val/
    employee/
    non_employee/
"""
import argparse
from pathlib import Path

from pydantic_core.core_schema import model_field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class BinaryEmployeeDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder: datasets.ImageFolder):
        self.image_folder = image_folder
        self.employee_idx = image_folder.class_to_idx["employee"]
        self.non_employee_idx = image_folder.class_to_idx["non_employee"]

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        image, label = self.image_folder[index]
        binary_label = 1 if label == self.employee_idx else 0
        return image, binary_label


def build_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 1)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", default="employee_classifier.pt", type=Path)
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--image-size", default=224, type=int)
    args = parser.parse_args()

    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(args.dataset_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(args.dataset_dir / "val", transform=eval_tfms)
    assert set(train_ds.class_to_idx.keys()) == {"employee", "non_employee"}, (
        f"Clases encontradas: {train_ds.class_to_idx}. Se espera employee y non_employee."
    )

    train_loader = DataLoader(
        BinaryEmployeeDataset(train_ds), batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        BinaryEmployeeDataset(val_ds), batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"[{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": "   ",
                    "model_state_dict": model.state_dict(),
                    "classes": {"employee": 1, "non_employee": 0},
                    "image_size": args.image_size,
                    "best_val_acc": best_val_acc,
                },
                args.output,
            )
            print(f"Nuevo mejor checkpoint guardado en: {args.output}")


if __name__ == "__main__":
    main()
