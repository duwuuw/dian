import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CNN

torch.manual_seed(2023)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 并没有加入数据增强，只拿transform做了微量处理
model_big_kernel = CNN(input_channels = 1, kernel_size = 7).to(device)
model_small_kernel = CNN(input_channels = 1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_big = optim.AdamW(model_big_kernel.parameters(), lr = 0.001,weight_decay = 3e-4)
scheduler_big = optim.lr_scheduler.StepLR(optimizer_big, step_size = 5, gamma = 0.875)
optimizer_small = optim.AdamW(model_small_kernel.parameters(), lr = 0.001,weight_decay = 3e-4)
scheduler_small = optim.lr_scheduler.StepLR(optimizer_small, step_size = 5, gamma = 0.875)
epochs = 30

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
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)


def train_one_epoch(model,loader,optimizer,criterion,device,scheduler):
    model.train()
    total_loss = 0.0
    for images,labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        optimizer.zero_grad()

    scheduler.step()
    return total_loss

def eval(model,loader,criterion,device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    for images,labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits,labels)
        total_loss += loss.item()
        total_correct += (logits.argmax(dim = 1) == labels).sum().item()
    return total_loss,total_correct


# ---------- 存指标 + 画图 ----------
def save_csv(out, rows):
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(out, rows):
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
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)


def train(model,loader,optimizer,criterion,device,scheduler):
    # 两个模型分开存，用卷积核大小区分
    out = OUT / f"kernel{model.conv1.kernel_size[0]}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_one_epoch(model,loader,optimizer,criterion,device,scheduler)

        # eval() 返回 (loss 累加, 预测对的张数)。
        # 它累加的是每个 batch 的平均 loss，所以要除以 batch 数才是一轮的 loss；
        # 准确率才是除以图片总数。
        train_loss_sum,train_correct = eval(model,loader,criterion,device)
        val_loss_sum,val_correct = eval(model,test_loader,criterion,device)

        rows.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss_sum / len(loader),
            "train_acc": train_correct / len(loader.dataset),
            "val_loss": val_loss_sum / len(test_loader),
            "val_acc": val_correct / len(test_loader.dataset),
        })
        save_csv(out, rows)     # 每轮都存，中途断了也不丢已经跑的
        plot_curves(out, rows)  # 每轮重画，训练中就能打开看

        if rows[-1]["val_acc"] > best_acc:
            best_acc = rows[-1]["val_acc"]
            torch.save(model.state_dict(), out / "best.pth")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={rows[-1]['train_loss']:.4f} train_acc={rows[-1]['train_acc']:.4f} | "
            f"val_loss={rows[-1]['val_loss']:.4f} val_acc={rows[-1]['val_acc']:.4f} | "
            f"lr={rows[-1]['lr']:.6f}"
        )

    print(f"Best val acc: {best_acc:.4f}")
    print(f"指标 -> {out / 'metrics.csv'}")
    print(f"曲线 -> {out / 'curves.png'}")
    print(f"权重 -> {out / 'best.pth'}")
    return rows


if __name__ == "__main__":
    train(model_big_kernel,train_loader,optimizer_big,criterion,device,scheduler_big)
    train(model_small_kernel,train_loader,optimizer_small,criterion,device,scheduler_small)
