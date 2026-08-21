import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from dataset import PizzaDataset, load_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 15
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-4
MODELS_DIR = Path(__file__).parent / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def evaluate_full(model, loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\nClassification report (validation set):")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


def main():
    split = load_split()
    classes = split["classes"]

    train_ds = PizzaDataset(split["train"], transform=train_transform)
    val_ds = PizzaDataset(split["val"], transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / "resnet18_pizza.pt"

    def maybe_save(val_acc):
        nonlocal best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, ckpt_path)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    # Phase 1: train only the new head, backbone frozen
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=HEAD_LR)

    print("\n=== Phase 1: training head (backbone frozen) ===")
    for epoch in range(1, HEAD_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        print(f"[head {epoch}/{HEAD_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-t0:.1f}s)")
        maybe_save(val_acc)

    # Phase 2: unfreeze everything, fine-tune at a lower LR
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)

    print("\n=== Phase 2: fine-tuning full network ===")
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        print(f"[finetune {epoch}/{FINE_TUNE_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-t0:.1f}s)")
        maybe_save(val_acc)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # Reload best checkpoint for the final report
    best = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(best["state_dict"])
    evaluate_full(model, val_loader, classes)


if __name__ == "__main__":
    main()
