"""数据：配对扫描 -> 清洗 -> 划分 -> 增强 -> Dataset。

本数据集结构:
    <root>/<日期>/dataset/input/*.jpg    拍摄的试卷（有手写、有阴影）
    <root>/<日期>/dataset/output/*.jpg   清理后的同一页（无手写、亮度归一）

配对是按「文件名主干」对应的，input 与 output 几何已对齐（ECC 残差 < 3px），
主要差异是亮度/阴影归一化 + 手写墨迹，所以可以直接做像素级回归。

用法:
    python data.py --root data/deli --out splits.json      # 清洗 + 划分
    python data.py --root data/deli --list                 # 只看目录结构
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
INPUT_DIRS = {"input", "inputs", "src", "source", "handwritten", "hw", "noisy", "before", "in"}
TARGET_DIRS = {"output", "outputs", "target", "clean", "gt", "label", "printed", "after", "out"}


# ================================================================ IO
def read_image(path: Path, downscale: int = 1) -> Optional[np.ndarray]:
    """读图为 RGB uint8。fromfile+imdecode 兼容中文路径；
    downscale>1 时用 JPEG 的 DCT 缩放解码（IMREAD_REDUCED_*），比全解码后再 resize 快 4~8 倍。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}.get(downscale,
                                                                                    cv2.IMREAD_COLOR)
        img = cv2.imdecode(buf, flag)
    except Exception:
        return None
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path: Path, rgb: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb


# ================================================================ 扫描配对
@dataclass
class Pair:
    input: str
    target: str
    page: str


def scan_pairs(root: Path) -> Tuple[List[Pair], str]:
    """按语义目录名配对。key 保留语义目录之前的相对路径，避免不同日期目录同名文件互相覆盖。"""
    ins: Dict[str, Path] = {}
    tgts: Dict[str, Path] = {}
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()):
        parts = list(f.relative_to(root).parts)
        dirs = [p.lower() for p in parts[:-1]]
        hit_in = next((i for i in range(len(dirs) - 1, -1, -1) if dirs[i] in INPUT_DIRS), None)
        hit_tg = next((i for i in range(len(dirs) - 1, -1, -1) if dirs[i] in TARGET_DIRS), None)
        if hit_in is not None and hit_tg is None:
            ins["/".join(parts[: hit_in + 1] + [f.stem])] = f
        elif hit_tg is not None and hit_in is None:
            tgts["/".join(parts[: hit_tg + 1] + [f.stem])] = f
    # 语义目录名不同（input vs output），用「语义目录之前的路径 + 主干」对齐
    norm = lambda k: "/".join(k.split("/")[:-2] + [k.split("/")[-1]])
    ins = {norm(k): v for k, v in ins.items()}
    tgts = {norm(k): v for k, v in tgts.items()}
    keys = sorted(set(ins) & set(tgts))
    pairs = [Pair(str(ins[k]), str(tgts[k]), k.replace("/", "_")) for k in keys]
    layout = "dir(input/output)" if pairs else "unknown"
    return pairs, layout


# ================================================================ 清洗
def _dhash(g: np.ndarray, size: int = 8) -> int:
    s = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    v = 0
    for b in (s[:, 1:] > s[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v


def clean_pairs(
    pairs: Sequence[Pair],
    downscale: int = 1,
    min_side: int = 96,
    max_aspect: float = 8.0,
    blank_std: float = 4.0,
    min_ink: float = 0.002,
    dedup: int = 2,
) -> List[Pair]:
    """剔除退化样本。关键是第 3、4 条：GT 若是空白页，等于教网络"清空整页"。"""
    kept, hashes = [], []
    drop = {"解码失败": 0, "过小": 0, "极端长宽比": 0, "输入空白": 0, "GT空白": 0,
            "GT无墨迹": 0, "重复": 0, "尺寸不符": 0}

    for p in pairs:
        a = read_image(Path(p.input), downscale)
        b = read_image(Path(p.target), downscale)
        if a is None or b is None:
            drop["解码失败"] += 1
            continue
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        if min(ha, wa) < min_side or min(hb, wb) < min_side:
            drop["过小"] += 1
            continue
        if max(ha, wa) / max(1, min(ha, wa)) > max_aspect or max(hb, wb) / max(1, min(hb, wb)) > max_aspect:
            drop["极端长宽比"] += 1
            continue
        if abs(ha / wa - hb / wb) > 0.06:
            drop["尺寸不符"] += 1
            continue

        ga, gb = gray(a), gray(b)
        if float(ga.std()) < blank_std:
            drop["输入空白"] += 1
            continue
        if float(gb.std()) < blank_std:
            drop["GT空白"] += 1
            continue
        if float((gb < 128).mean()) < min_ink:      # GT 必须残留印刷文字
            drop["GT无墨迹"] += 1
            continue

        h = _dhash(gb)
        if any(bin(h ^ x).count("1") <= dedup for x in hashes):
            drop["重复"] += 1
            continue
        hashes.append(h)
        kept.append(p)

    total = len(pairs)
    detail = "  ".join(f"{k}={v}" for k, v in drop.items() if v)
    print(f"[清洗] {total} 对 -> 保留 {len(kept)} 对 ({len(kept)/max(1,total)*100:.1f}%)  丢弃: {detail}")
    return kept


# ================================================================ 划分
def split_pairs(pairs: Sequence[Pair], ratios=(0.8, 0.1, 0.1), seed: int = 3407):
    groups: Dict[str, List[Pair]] = {}
    for p in pairs:
        groups.setdefault(p.page, []).append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_tr, n_va = int(len(keys) * ratios[0]), int(len(keys) * ratios[1])
    part = {"train": keys[:n_tr], "val": keys[n_tr: n_tr + n_va], "test": keys[n_tr + n_va:]}
    return {k: [p for key in v for p in groups[key]] for k, v in part.items()}


def save_split(splits, path: Path) -> None:
    payload = {k: [asdict(p) for p in v] for k, v in splits.items()}
    payload["_counts"] = {k: len(v) for k, v in splits.items()}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_split(path: Path) -> Dict[str, List[Pair]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: [Pair(**x) for x in v] for k, v in d.items() if not k.startswith("_")}


# ================================================================ 增强
class DocAugment:
    """几何变换对 (输入, GT) 同步施加；光度退化只作用于输入。

    这里的光度退化是刻意加重的：本数据集的 GT 是"亮度归一 + 去手写"的版本，
    所以网络必须学会消除光照不均、阴影、噪声、模糊、JPEG 伪影，
    而这些正是真实拍照试卷里的常见退化。
    """

    def __init__(self, rot=2.5, scale=(0.92, 1.08), persp=0.012, bright=0.20,
                 contrast=0.20, gamma=(0.65, 1.5), noise=0.035,
                 blur_p=0.35, jpeg_p=0.35, illum_p=0.6):
        self.rot, self.scale, self.persp = rot, scale, persp
        self.bright, self.contrast, self.gamma = bright, contrast, gamma
        self.noise, self.blur_p, self.jpeg_p, self.illum_p = noise, blur_p, jpeg_p, illum_p

    def _geom(self, a, b):
        h, w = a.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-self.rot, self.rot),
                                    random.uniform(*self.scale))
        m[0, 2] += random.uniform(-0.02, 0.02) * w
        m[1, 2] += random.uniform(-0.02, 0.02) * h
        if self.persp > 0:
            d = self.persp * min(h, w)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = src + np.float32([[random.uniform(-d, d), random.uniform(-d, d)] for _ in range(4)])
            m = cv2.getPerspectiveTransform(src, dst) @ np.vstack([m, [0, 0, 1]])
        return (cv2.warpPerspective(a, m, (w, h), borderMode=cv2.BORDER_REPLICATE),
                cv2.warpPerspective(b, m, (w, h), borderMode=cv2.BORDER_REPLICATE))

    def _photo(self, x):
        h, w = x.shape[:2]
        if random.random() < self.illum_p:                      # 光照不均 / 阴影
            f = np.random.rand(max(2, h // 24), max(2, w // 24)).astype(np.float32) * 0.30 + 0.76
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
            if random.random() < 0.5:
                f = cv2.GaussianBlur(f, (0, 0), sigmaX=max(h, w) / 18)
            x = x * f[..., None]
        x = x * (1 + random.uniform(-self.contrast, self.contrast)) + random.uniform(-self.bright, self.bright)
        if random.random() < 0.6:                               # 曝光 / 墨迹浓淡
            x = np.clip(x, 0, 1) ** random.uniform(*self.gamma)
        if random.random() < self.blur_p:                       # 失焦 / 运动模糊
            if random.random() < 0.5:
                k = random.choice([3, 5])
                x = cv2.GaussianBlur(x, (k, k), 0)
            else:
                k = random.choice([5, 7])
                kern = np.zeros((k, k), np.float32)
                kern[k // 2, :] = 1.0
                kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D(
                    (k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1.0), (k, k))
                x = cv2.filter2D(x, -1, kern / max(kern.sum(), 1e-6))
        if self.noise > 0:                                      # 传感器噪声
            x = x + np.random.randn(*x.shape).astype(np.float32) * random.uniform(0, self.noise)
        if random.random() < self.jpeg_p:                       # 压缩伪影
            enc = cv2.imencode(".jpg", cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(35, 90)])[1]
            x = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        return x

    def __call__(self, a, b):
        a, b = self._geom(a, b)
        a = self._photo(a)
        return (np.clip(a, 0, 1).astype(np.float32), np.clip(b, 0, 1).astype(np.float32))


# ================================================================ Dataset
class PairedDataset(Dataset):
    """train: 随机裁剪 patch，按手写密度偏置采样位置；eval: 确定性中心裁剪。

    性能: 解码用 DCT 缩放（downscale），并按字节上限缓存解码结果与手写密度图，
    避免每个 step 重复解码同一张大图。实测这一步比模型前向本身贵得多。
    """

    def __init__(self, pairs, patch=256, train=True, augment=None, downscale=1,
                 bias=0.6, repeat=1, limit=None, cache_mb=1500):
        self.pairs = list(pairs)[:limit] if limit else list(pairs)
        if not self.pairs:
            raise ValueError("数据集为空")
        self.patch, self.train, self.aug = patch, train, augment
        self.downscale, self.bias, self.repeat = downscale, bias, repeat
        self._cache: Dict[str, Tuple] = {}
        self._cache_bytes, self._cache_max = 0, cache_mb * 1024 * 1024

    def __len__(self):
        return len(self.pairs) * (self.repeat if self.train else 1)

    def _load(self, i):
        p = self.pairs[i % len(self.pairs)]
        hit = self._cache.get(p.input)
        if hit is not None:
            return hit
        a, b = read_image(Path(p.input), self.downscale), read_image(Path(p.target), self.downscale)
        if a is None or b is None:
            a = np.full((256, 256, 3), 255, np.uint8)
            b = a.copy()
        if b.shape[:2] != a.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        # 手写密度图：|输入 - GT| 的模糊图，作为裁剪位置的概率图
        d = cv2.GaussianBlur(np.abs(gray(a).astype(np.float32) - gray(b).astype(np.float32)),
                             (0, 0), sigmaX=max(8, min(a.shape[:2]) / 16))
        item = (a, b, d, cv2.integral(d))
        nbytes = sum(x.nbytes for x in item)
        if self._cache_bytes + nbytes < self._cache_max:
            self._cache[p.input] = item
            self._cache_bytes += nbytes
        return item

    def _crop(self, a, b, d, ii):
        h, w = a.shape[:2]
        ph = pw = self.patch
        if h < ph or w < pw:                 # 比 patch 还小的图（试卷条带）：边界填充后再裁
            bt, rt = max(0, ph - h), max(0, pw - w)
            a = cv2.copyMakeBorder(a, 0, bt, 0, rt, cv2.BORDER_REPLICATE)
            b = cv2.copyMakeBorder(b, 0, bt, 0, rt, cv2.BORDER_REPLICATE)
            d = cv2.copyMakeBorder(d, 0, bt, 0, rt, cv2.BORDER_REPLICATE)
            ii = cv2.integral(d)
            h, w = a.shape[:2]
        if not self.train:
            y0, x0 = (h - ph) // 2, (w - pw) // 2
        elif self.bias > 0 and random.random() < self.bias and d.max() > 1e-6:
            ys = np.linspace(0, max(0, h - ph), num=min(32, max(1, h - ph + 1))).astype(int)
            xs = np.linspace(0, max(0, w - pw), num=min(32, max(1, w - pw + 1))).astype(int)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            s = (ii[yy + ph, xx + pw] - ii[yy, xx + pw] - ii[yy + ph, xx] + ii[yy, xx]).flatten()
            p = s.astype(np.float64) + 1e-6
            k = int(np.random.choice(len(p), p=p / p.sum()))
            y0, x0 = int(yy.flatten()[k]), int(xx.flatten()[k])
        else:
            y0, x0 = random.randint(0, h - ph), random.randint(0, w - pw)
        return a[y0:y0 + ph, x0:x0 + pw], b[y0:y0 + ph, x0:x0 + pw]

    def __getitem__(self, i):
        a, b, d, ii = self._load(i)
        a = a.astype(np.float32) / 255.0
        b = b.astype(np.float32) / 255.0
        a, b = self._crop(a, b, d, ii)
        if self.train and self.aug is not None:
            a, b = self.aug(a, b)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return {"input": t(a), "target": t(b), "name": self.pairs[i % len(self.pairs)].page}


# ================================================================ CLI
def build_splits(root: Path, out: Path, downscale=1, ratios=(0.8, 0.1, 0.1),
                 seed=3407, limit=None, min_side=96, blank_std=4.0) -> Dict[str, List[Pair]]:
    pairs, layout = scan_pairs(root)
    print(f"[扫描] {root}  布局={layout}  配对={len(pairs)}")
    if not pairs:
        raise SystemExit("未识别到配对结构，请用 --list 查看目录分布")
    pairs = clean_pairs(pairs, downscale, min_side=min_side, blank_std=blank_std)
    if limit:
        pairs = random.Random(seed).sample(pairs, min(limit, len(pairs)))
        print(f"[限流] 保留 {len(pairs)} 对")
    splits = split_pairs(pairs, ratios, seed)
    save_split(splits, out)
    print(f"[划分] train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}"
          f"  ->  {out}")
    return splits


def main():
    ap = argparse.ArgumentParser(description="数据清洗与划分")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("splits.json"))
    ap.add_argument("--downscale", type=int, default=2, choices=[1, 2, 4, 8])
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, default=None, help="只用 N 对（快速验证流程）")
    ap.add_argument("--min_side", type=int, default=96)
    ap.add_argument("--blank_std", type=float, default=4.0)
    ap.add_argument("--list", action="store_true", help="只打印目录分布")
    a = ap.parse_args()

    if a.list:
        from collections import Counter
        files = [p for p in a.root.rglob("*") if p.suffix.lower() in IMG_EXT]
        print(f"共 {len(files)} 个图像文件")
        c = Counter(p.relative_to(a.root).parts[0] for p in files)
        print("顶层目录:", c.most_common(20))
        pairs, layout = scan_pairs(a.root)
        print(f"布局={layout}  配对={len(pairs)}")
        for p in pairs[:3]:
            print(f"  例: in={Path(p.input).name}  gt={Path(p.target).name}")
        return

    build_splits(a.root, a.out, a.downscale, tuple(a.ratios), a.seed, a.limit,
                 a.min_side, a.blank_std)


if __name__ == "__main__":
    main()
