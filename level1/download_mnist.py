"""下载 MNIST 数据集到本项目内的 data/ 目录。

用法:
    python download_mnist.py
"""

from pathlib import Path

from torchvision import datasets, transforms

# 数据统一放在项目根目录下的 data/ 里，跟代码分开
DATA_ROOT = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # download=True 时，如果 data/MNIST/raw 下已经有文件就不会重复下载
    for train in (True, False):
        datasets.MNIST(
            root=str(DATA_ROOT),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    split = {True: "train", False: "test"}
    for train in (True, False):
        ds = datasets.MNIST(root=str(DATA_ROOT), train=train, download=False)
        print(f"{split[train]:>5}: {len(ds)} images")

    print(f"\n保存位置: {DATA_ROOT / 'MNIST'}")


if __name__ == "__main__":
    main()
