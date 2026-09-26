"""推理：对整页试卷做手写擦除。

整页图远大于训练 patch，直接整图前向会有显存和统计不一致（边缘发虚）两个问题，
所以按 tile 滑窗推理，用余弦权重羽化拼接，块间无可见接缝。

产物:
    <out>/clean/*.png     擦除后的干净图
    <out>/compare/*.png   输入 | 预测 | GT 三联对比图
    <out>/report.json     每张图的 PSNR/SSIM 与全白·全黑·墨水丢失检测

用法:
    # 推荐：直接吃 splits.json，自动配对 input/GT 并算指标
    python inference.py --ckpt runs/xxx/best.pth --splits splits.json --split test \
                        --output pred --limit 30
    # 或指定目录 / 单张图
    python inference.py --ckpt runs/xxx/best.pth --input data/deli/20250213/dataset/input \
                        --gt data/deli/20250213/dataset/output --output pred
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from data import IMG_EXT, load_split, read_image, write_image
from model import build_model, blank_report, psnr_np, ssim_np


def load_model(ckpt: str, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    m = build_model(a.get("arch", "unet"), 3, a.get("base_ch", 32), a.get("depth", 4),
                    not a.get("no_residual", False), a.get("attn", False),
                    dim=a.get("dim", 256), blocks=a.get("blocks", 6),
                    heads=a.get("heads", 8)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    print(f"[模型] {ckpt} | step={ck.get('step')} | arch={a.get('arch', 'unet')} "
          f"base={a.get('base_ch')} depth={a.get('depth')} "
          f"| 训练 best PSNR={ck.get('best_psnr', float('nan')):.2f} | downscale=1/{a.get('downscale', 1)}")
    return m, ck, a.get("downscale", 1)


def _window(tile, overlap, device):
    w = torch.ones(tile, dtype=torch.float32)
    if overlap > 0:
        r = torch.linspace(0, 1, overlap + 2)[1:-1]
        r = 0.5 - 0.5 * torch.cos(np.pi * r)
        w[:overlap], w[-overlap:] = r, r.flip(0)
    return (w[:, None] * w[None, :]).clamp_min(1e-3).to(device)


@torch.no_grad()
def tiled_infer(model, img, tile=512, overlap=64, device=torch.device("cuda")):
    """img: HWC float32 [0,1] -> 同尺寸擦除结果。"""
    h, w = img.shape[:2]
    tile = min(tile, max(64, min(h, w)))
    overlap = min(overlap, tile // 2)
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).to(device)
    amp = device.type == "cuda"

    if h <= tile and w <= tile:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        return out.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    stride = tile - overlap
    ys = sorted(set(list(range(0, max(1, h - tile + 1), stride)) + [max(0, h - tile)]))
    xs = sorted(set(list(range(0, max(1, w - tile + 1), stride)) + [max(0, w - tile)]))
    coords = [(y, xx) for y in ys for xx in xs]

    win = _window(tile, overlap, device)
    acc = torch.zeros(3, h, w, device=device)
    wsum = torch.zeros(1, h, w, device=device)
    for i in range(0, len(coords), 4):
        ch = coords[i:i + 4]
        batch = torch.stack([x[0, :, y:y + tile, xx:xx + tile] for y, xx in ch])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(batch)
        pred = pred.float().clamp(0, 1)
        for k, (y, xx) in enumerate(ch):
            acc[:, y:y + tile, xx:xx + tile] += pred[k] * win
            wsum[:, y:y + tile, xx:xx + tile] += win
    return (acc / wsum.clamp_min(1e-6)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def make_compare(inp, pred, gt=None):
    panels = [inp, pred] + ([gt] if gt is not None else [])
    labels = ["INPUT (with handwriting)", "PRED (erased)"] + (["GT (clean)"] if gt is not None else [])
    bar, gap = 26, 8
    H = max(p.shape[0] for p in panels)
    W = sum(p.shape[1] for p in panels) + gap * (len(panels) + 1)
    c = np.full((H + bar + gap, W, 3), 1.0, np.float32)
    x = gap
    for p, lb in zip(panels, labels):
        c[bar:bar + p.shape[0], x:x + p.shape[1]] = p
        cv2.putText(c, lb, (x + 4, bar - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        x += p.shape[1] + gap
    return c


def main():
    ap = argparse.ArgumentParser(description="手写擦除推理")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--splits", type=str, default=None,
                    help="用 splits.json 里指定的划分（推荐，自动配对 input/GT）")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--input", type=str, default=None, help="图片文件或目录")
    ap.add_argument("--gt", type=str, default=None, help="干净图目录（可选，用于算指标）")
    ap.add_argument("--output", type=str, default="pred")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_compare", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck, downscale = load_model(a.ckpt, device)

    # 组装待推理清单: [(输入路径, GT 路径或 None), ...]
    if a.splits:
        pairs = load_split(Path(a.splits))[a.split]
        pairs = pairs[:a.limit] if a.limit else pairs
        todo = [(Path(p.input), Path(p.target)) for p in pairs]
        print(f"[输入] {a.splits} 的 {a.split} 划分: {len(todo)} 对")
    else:
        if not a.input:
            raise SystemExit("需要 --splits 或 --input 之一")
        src = Path(a.input)
        files = sorted(p for p in ([src] if src.is_file() else src.rglob("*"))
                       if p.suffix.lower() in IMG_EXT)
        files = files[:a.limit] if a.limit else files
        gd = Path(a.gt) if a.gt else None
        todo = [(f, (gd / f.name if gd else None)) for f in files]
    if not todo:
        raise SystemExit("没有待推理的图片")

    out = Path(a.output)
    reports, t0 = [], time.time()

    for i, (f, gt_path) in enumerate(todo, 1):
        u8 = read_image(f, downscale)
        if u8 is None:
            print(f"  [跳过] 无法解码: {f.name}")
            continue
        inp = u8.astype(np.float32) / 255.0

        t = time.time()
        pred = tiled_infer(model, inp, a.tile, a.overlap, device)
        dt = time.time() - t

        gt = None
        if gt_path is not None:
            cand = gt_path if gt_path.exists() else next(iter(gt_path.parent.rglob(gt_path.name)), None)
            if cand is not None:
                g = read_image(Path(cand), downscale)
                # 原图尺寸相同，但 cv2 的 IMREAD_REDUCED_* 对两张 JPEG 可能解出差 1 像素的尺寸
                if g is not None and g.shape[:2] != u8.shape[:2] and \
                        abs(g.shape[0] - u8.shape[0]) <= 2 and abs(g.shape[1] - u8.shape[1]) <= 2:
                    g = cv2.resize(g, (u8.shape[1], u8.shape[0]), interpolation=cv2.INTER_AREA)
                if g is not None and g.shape == u8.shape:
                    gt = g.astype(np.float32) / 255.0

        rec = {"name": f.name, "size": [u8.shape[1], u8.shape[0]], "infer_sec": round(dt, 2)}
        rec.update(blank_report(pred, f.name, gt=gt, inp=inp))
        if gt is not None:
            rec["psnr_in"], rec["ssim_in"] = round(psnr_np(inp, gt), 3), round(ssim_np(inp, gt), 4)
            rec["psnr"], rec["ssim"] = round(psnr_np(pred, gt), 3), round(ssim_np(pred, gt), 4)
        reports.append(rec)

        write_image(out / "clean" / f.name, (pred * 255).astype(np.uint8))
        if not a.no_compare:
            write_image(out / "compare" / f.name, (make_compare(inp, pred, gt) * 255).astype(np.uint8))

        msg = f"  [{i}/{len(todo)}] {f.name}  {u8.shape[1]}x{u8.shape[0]}  {dt:.2f}s"
        if "psnr" in rec:
            msg += f"  PSNR {rec['psnr_in']:.2f} -> {rec['psnr']:.2f} dB | SSIM {rec['ssim_in']:.3f} -> {rec['ssim']:.3f}"
        if not rec["ok"]:
            msg += f"  [异常 {rec['flags']}]"
        print(msg)

    (out / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 68)
    ev = [r for r in reports if "psnr" in r]
    if ev:
        print(f"[指标] 平均 PSNR {np.mean([r['psnr_in'] for r in ev]):.2f} -> "
              f"{np.mean([r['psnr'] for r in ev]):.2f} dB   (提升的图 "
              f"{sum(1 for r in ev if r['psnr'] > r['psnr_in'])}/{len(ev)})")
        print(f"[指标] 平均 SSIM {np.mean([r['ssim_in'] for r in ev]):.4f} -> "
              f"{np.mean([r['ssim'] for r in ev]):.4f}")
        keep = [r["ink_keep"] for r in ev if "ink_keep" in r]
        rem = [r["removed_frac"] for r in ev if "removed_frac" in r]
        if keep:
            print(f"[指标] 墨迹保留率 {np.mean(keep):.3f}±{np.std(keep):.3f} (1.0=印刷体完整保留)")
        if rem:
            print(f"[指标] 擦除率     {np.mean(rem):.3f}±{np.std(rem):.3f} (1.0=输入与GT差异被完全消除)")
    bad = [r for r in reports if not r["ok"]]
    print(f"[健康检查] {len(reports)} 张中异常 {len(bad)} 张 ({len(bad)/max(1,len(reports))*100:.1f}%)"
          f" | mean={np.mean([r['mean'] for r in reports]):.3f} std={np.mean([r['std'] for r in reports]):.3f}")
    for r in bad[:10]:
        print(f"   [异常] {r['name']}: {r['flags']} mean={r['mean']:.3f} std={r['std']:.3f}")
    print(f"[完成] {len(reports)} 张，用时 {(time.time()-t0)/60:.1f} 分钟 -> {out}/clean/")
    if not bad:
        print("[结论] 无全白/全黑/墨水丢失，可视化结果正常。")


if __name__ == "__main__":
    main()
