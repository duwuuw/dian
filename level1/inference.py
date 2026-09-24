"""
声明：
作图是ai写的
推理和评测是我自己写的

"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP

CKPT = Path("runs/baseline/best.pth")
OUT = CKPT.parent / "inference"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 数据 ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ---------- 推理 ----------
criterion = nn.CrossEntropyLoss()

running_loss = 0.0
correct = 0
total = 0
cm = torch.zeros(10, 10, dtype=torch.long)   # 行=真值，列=预测
preds_rows = []

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images.to(device))

        running_loss += criterion(logits, labels.to(device)).item() * images.size(0)

        prob = logits.softmax(dim=1)
        conf, preds = prob.max(dim=1)
        preds = preds.cpu()
        conf = conf.cpu()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i, (t, p, c) in enumerate(zip(labels.tolist(), preds.tolist(), conf.tolist())):
            cm[t][p] += 1
            preds_rows.append({"index": len(preds_rows), "true": t, "pred": p,
                               "correct": int(t == p), "confidence": round(c, 6)})

test_loss = running_loss / total
test_acc = correct / total
per_class_acc = (cm.diag() / cm.sum(1)).tolist()

print(f"测试集 loss={test_loss:.4f}  acc={test_acc:.4f}  ({correct}/{total})")
for digit, acc in enumerate(per_class_acc):
    print(f"  {digit}: {acc:.4f}")

# ---------- 存指标 ----------
metrics = {
    "ckpt": str(CKPT),
    "loss": test_loss,
    "acc": test_acc,
    "num_samples": total,
    "num_wrong": total - correct,
    "per_class_acc": per_class_acc,
    "confusion_matrix": cm.tolist(),
}
with open(OUT / "inference_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(OUT / "predictions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=preds_rows[0].keys())
    writer.writeheader()
    writer.writerows(preds_rows)


# ---------- 画图（ai 写的）----------
def plot_confusion_matrix(path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="count")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#333333")

    ax.set(xlabel="pred", ylabel="true", title=f"Confusion matrix (acc={test_acc:.4f})")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_acc(path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), per_class_acc, color="steelblue")

    for i, acc in enumerate(per_class_acc):
        ax.text(i, acc + 0.002, f"{acc:.3f}", ha="center", fontsize=8)

    ax.axhline(test_acc, color="black", ls="--", lw=1, label=f"overall {test_acc:.4f}")
    ax.set(xlabel="digit", ylabel="acc", title="Per-class accuracy")
    ax.set_xticks(range(10))
    ax.set_ylim(min(per_class_acc) - 0.05, 1.02)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


plot_confusion_matrix(OUT / "confusion_matrix.png")
plot_per_class_acc(OUT / "per_class_acc.png")

print(f"指标 -> {OUT / 'inference_metrics.json'}")
print(f"明细 -> {OUT / 'predictions.csv'}")
print(f"混淆矩阵 -> {OUT / 'confusion_matrix.png'}")
print(f"每类准确率 -> {OUT / 'per_class_acc.png'}")
