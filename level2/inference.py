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

from model import CNN

CKPT_DIR = Path("runs/baseline")
OUT = Path("runs/inference")
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_model(ckpt):
    """从权重里反推结构：conv1 的形状是 (32, input_channels, k, k)，fc2 是类别数。"""
    state = torch.load(ckpt, map_location=device)
    model = CNN(
        input_channels = state["conv1.weight"].shape[1],
        num_classes = state["fc2.weight"].shape[0],
        kernel_size = state["conv1.weight"].shape[2],
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run(model, loader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    cm = torch.zeros(10, 10, dtype=torch.long)   # 行=真值，列=预测
    rows = []

    for images, labels in loader:
        logits = model(images.to(device))

        running_loss += criterion(logits, labels.to(device)).item() * images.size(0)

        prob = logits.softmax(dim=1)
        conf, preds = prob.max(dim=1)
        preds = preds.cpu()
        conf = conf.cpu()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for t, p, c in zip(labels.tolist(), preds.tolist(), conf.tolist()):
            cm[t][p] += 1
            rows.append({"index": len(rows), "true": t, "pred": p,
                         "correct": int(t == p), "confidence": round(c, 6)})

    return running_loss / total, correct / total, cm, rows


# ---------- 画图（ai 写的）----------
def plot_confusion_matrix(cm, test_acc, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="count")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#333333")

    ax.set(xlabel="pred", ylabel="true", title=f"{title}  (acc={test_acc:.4f})")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_acc(per_class_acc, test_acc, path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), per_class_acc, color="steelblue")

    for i, acc in enumerate(per_class_acc):
        ax.text(i, acc + 0.002, f"{acc:.3f}", ha="center", fontsize=8)

    ax.axhline(test_acc, color="black", ls="--", lw=1, label=f"overall {test_acc:.4f}")
    ax.set(xlabel="digit", ylabel="acc", title=f"Per-class accuracy  ({title})")
    ax.set_xticks(range(10))
    ax.set_ylim(min(per_class_acc) - 0.05, 1.02)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 推理 ----------
criterion = nn.CrossEntropyLoss()

for ckpt in sorted(CKPT_DIR.glob("kernel*/best.pth")):
    tag = ckpt.parent.name          # kernel7 / kernel3
    model = load_model(ckpt)

    test_loss, test_acc, cm, preds_rows = run(model, test_loader, criterion)
    per_class_acc = (cm.diag() / cm.sum(1)).tolist()

    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] kernel={model.conv1.kernel_size[0]}  "
          f"loss={test_loss:.4f}  acc={test_acc:.4f}  "
          f"({int(cm.diag().sum())}/{int(cm.sum())})")

    for digit, acc in enumerate(per_class_acc):
        print(f"    {digit}: {acc:.4f}")

    # 存指标
    with open(out / "inference_metrics.json", "w") as f:
        json.dump({
            "ckpt": str(ckpt),
            "kernel_size": model.conv1.kernel_size[0],
            "loss": test_loss,
            "acc": test_acc,
            "num_samples": int(cm.sum()),
            "num_wrong": int(cm.sum() - cm.diag().sum()),
            "per_class_acc": per_class_acc,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    with open(out / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=preds_rows[0].keys())
        writer.writeheader()
        writer.writerows(preds_rows)

    # 画图
    plot_confusion_matrix(cm.numpy(), test_acc, out / "confusion_matrix.png", tag)
    plot_per_class_acc(per_class_acc, test_acc, out / "per_class_acc.png", tag)

    print(f"    指标 -> {out / 'inference_metrics.json'}")
    print(f"    明细 -> {out / 'predictions.csv'}")
    print(f"    混淆矩阵 -> {out / 'confusion_matrix.png'}")
    print(f"    每类准确率 -> {out / 'per_class_acc.png'}")
