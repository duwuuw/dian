
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP

torch.manual_seed(2023)

EPOCHS = 30
OUT = Path("runs/baseline")
OUT.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

rows = []


def save_csv():
    with open(OUT / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path):
    epoch = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(epoch, [r["train_loss"] for r in rows], "-o", label="train")
    ax[0].plot(epoch, [r["val_loss"] for r in rows], "-s", label="val")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")

    ax[1].plot(epoch, [r["train_acc"] for r in rows], "-o", label="train")
    ax[1].plot(epoch, [r["val_acc"] for r in rows], "-s", label="val")
    ax[1].set(xlabel="epoch", ylabel="acc", title="Accuracy")

    for a in ax:
        a.grid(alpha=0.3)
        a.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 训练 ----------
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    scheduler.step()

    rows.append({
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })
    save_csv()                        # 每轮都存，中途断了也不丢已经跑的
    plot_curves(OUT / "curves.png")   # 每轮重画，训练中就能打开看

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), OUT / "best.pth")

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
        f"lr={rows[-1]['lr']:.6f}"
    )

print(f"Best val acc: {best_acc:.4f}")
print(f"指标 -> {OUT / 'metrics.csv'}")
print(f"曲线 -> {OUT / 'curves.png'}")
print(f"权重 -> {OUT / 'best.pth'}")
