"""训练：以 step 为单位，每 --eval_every 步输出一次 PSNR / SSIM 曲线。

产物 (runs/<时间戳>/):
    log.csv     每步的 loss 分项 / PSNR / SSIM / 耗时 / 费用
    curves.png  Loss + PSNR + SSIM 三联演化曲线
    cost.json   训练时长与费用明细
    best.pth    验证 PSNR 最优权重

典型用法:
    # 小数据快速验证流程（成本 < 0.2 元）
    python train.py --splits splits.json --limit_train 32 --max_steps 100 \
                    --eval_every 20 --patch 192 --tag quick

    # 正式训练
    python train.py --splits splits.json --max_steps 1000 --eval_every 20 \
                    --batch_size 8 --patch 256 --amp bf16 --tag run1000
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data import DocAugment, PairedDataset, build_splits, load_split
from model import RestorationLoss, build_model, ink_metrics, psnr, ssim

# 手写区误差分解的四项，训练中每步评估都算（AGENT.md 5.2）
INK_KEYS = ("ink_err", "bg_err", "ghost", "ghost_keep")


# ================================================================ 评估
@torch.no_grad()
def evaluate(model, loader, crit, device, dtype):
    model.eval()
    P, S, L, M = [], [], [], []
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            pred = model(x)
        pred = pred.float().clamp(0, 1)                 # 与推理时一致
        P.append(psnr(pred, y).mean().item())
        S.append(ssim(pred, y).mean().item())
        L.append(float(crit(pred, y, x)["total"]))
        mk = ink_metrics(pred, y, x)                    # PSNR 被光度归一化主导，必须单独看手写区
        if mk:
            M.append(mk)
    model.train()
    out = {"psnr": float(np.mean(P)), "ssim": float(np.mean(S)), "val_loss": float(np.mean(L))}
    for k in INK_KEYS:
        v = [m[k] for m in M if k in m]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


class _ListDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ================================================================ 曲线
class Curves:
    FIELDS = ["step", "epoch", "lr", "loss", "l1", "mse", "ssim_loss", "edge", "ghost_loss", "hf_loss",
              "vgg_loss",
              "val_loss", "psnr", "ssim", "ink_err", "bg_err", "ghost", "ghost_keep",
              "sec_per_step", "elapsed_min", "cost_yuan"]

    def __init__(self, out: Path, price: float, note: str):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.csv, self.png = out / "log.csv", out / "curves.png"
        self.price, self.note, self.rows = price, note, []

    def add(self, row):
        self.rows.append(row)

    def flush(self):
        with open(self.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)

    def plot(self, title=""):
        if not self.rows:
            return
        s = [r["step"] for r in self.rows]
        fig, ax = plt.subplots(4, 1, figsize=(9, 12.5), sharex=True)

        ax[0].plot(s, [r["loss"] for r in self.rows], lw=2, label="train total", color="#1f77b4")
        for k, c in [("l1", "#ff7f0e"), ("mse", "#2ca02c"), ("ssim_loss", "#d62728"),
                     ("edge", "#9467bd"), ("ghost_loss", "#8c564b"), ("hf_loss", "#e377c2")]:
            v = [r.get(k, np.nan) for r in self.rows]
            if any(np.isfinite(v)):
                ax[0].plot(s, v, lw=1, alpha=0.75, label=k, color=c)
        ax[0].plot(s, [r.get("val_loss", np.nan) for r in self.rows], lw=2, ls="--",
                   label="val total", color="#7f7f7f")
        ax[0].set(ylabel="Loss", yscale="log", title=title or "Training curves")
        ax[0].grid(alpha=0.3)
        ax[0].legend(fontsize=8)

        # 验证集只有十几张图、每张内容差异大，逐点曲线会很抖，
        # 所以叠一条滑动平均趋势线，原始点用细线画在下面
        for a, key, col, mk in [(ax[1], "psnr", "#2ca02c", "o-"), (ax[2], "ssim", "#d62728", "s-")]:
            v = np.asarray([r[key] for r in self.rows], dtype=float)
            a.plot(s, v, mk, ms=2, lw=0.6, alpha=0.35, color=col, label="raw")
            w = max(1, min(len(v) // 20, 25))
            if w > 1:
                k = np.ones(w) / w
                a.plot(s[w - 1:], np.convolve(v, k, mode="valid"), lw=2, color=col,
                       label=f"moving avg ({w} evals)")
            b = int(np.nanargmax(v))
            a.scatter([s[b]], [v[b]], s=70, facecolors="none", edgecolors="red", zorder=5,
                      label=f"best {v[b]:.4f}" if key == "ssim" else f"best {v[b]:.2f} dB")
            a.set_ylabel("PSNR (dB)" if key == "psnr" else "SSIM")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)

        # 第四联：手写区误差分解。PSNR 有 71% 的损失来自光度归一化，会被它主导；
        # 真正决定"手写擦干净没有"的是这两条线，验收也看它们。
        # 注：标签用英文——matplotlib 默认的 DejaVu Sans 没有中文字形，中文会画成空框
        for k, c, lb in [("ink_err", "#d62728", "ink-region |pred-GT|"),
                         ("bg_err", "#1f77b4", "background |pred-GT|"),
                         ("ghost", "#8c564b", "ghost (residual stroke in mask)")]:
            v = np.asarray([r.get(k, np.nan) for r in self.rows], dtype=float)
            if not np.isfinite(v).any():
                continue
            a = ax[3]
            a.plot(s, v, "o-", ms=2, lw=0.6, alpha=0.35, color=c)
            w = max(1, min(len(v) // 20, 25))
            if w > 1 and np.isfinite(v).all():
                a.plot(s[w - 1:], np.convolve(v, np.ones(w) / w, mode="valid"), lw=2, color=c, label=lb)
            else:
                a.plot(s, v, lw=2, color=c, label=lb)
        ax[3].set(ylabel="Error (gray 0~1)", yscale="log")
        ax[3].grid(alpha=0.3)
        b = int(np.nanargmin([r.get("ink_err", np.nan) for r in self.rows]))
        ax[3].scatter([s[b]], [self.rows[b].get("ink_err", np.nan)], s=70, facecolors="none",
                      edgecolors="red", zorder=5,
                      label=f"best ink_err {self.rows[b].get('ink_err', float('nan')):.4f}")
        ax[3].legend(fontsize=8)
        ax[3].set_xlabel("Training step")

        last = self.rows[-1]
        fig.suptitle(f"{title} | {last['elapsed_min']:.1f} min | {last['sec_per_step']:.2f} s/step "
                     f"| {last['cost_yuan']:.2f} CNY @ {self.price} CNY/h", fontsize=9, y=0.995)
        fig.tight_layout()
        fig.savefig(self.png, dpi=130)
        plt.close(fig)


# ================================================================ 主流程
def parse_args():
    p = argparse.ArgumentParser(description="U-Net 手写擦除训练")
    p.add_argument("--splits", type=str, default="splits.json")
    p.add_argument("--data_root", type=str, default="data/deli", help="splits 不存在时自动划分")
    p.add_argument("--downscale", type=int, default=2, choices=[1, 2, 4, 8])
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--eval_patch", type=int, default=None)
    p.add_argument("--eval_images", type=int, default=12)
    p.add_argument("--limit_train", type=int, default=None)
    p.add_argument("--limit_data", type=int, default=None, help="自动划分时只用 N 对")
    # 模型
    p.add_argument("--arch", type=str, default="unet", choices=["unet", "dit"],
                   help="unet=卷积 U-Net；dit=DiT 风格 Transformer（直接回归，非扩散采样）")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--attn", action="store_true")
    # 仅 --arch dit 使用
    p.add_argument("--dim", type=int, default=256, help="Transformer 隐藏维度")
    p.add_argument("--blocks", type=int, default=6, help="Transformer block 数")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--no_rope", action="store_true",
                   help="DiT 关闭 RoPE，退回可学习位置编码+推理插值（消融用；默认开 RoPE）")
    # 损失
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_mse", type=float, default=0.0)
    p.add_argument("--w_ssim", type=float, default=0.2)
    p.add_argument("--w_edge", type=float, default=0.2)
    p.add_argument("--mask_weight", type=float, default=5.0,
                   help=">0 时按手写代理掩码给 L1 加权。手写只占少数像素，权重不够就会被"
                        "大面积的光照归一化差异淹没（实测 0.5 时掩码覆盖率 95%%，等于没加权）")
    p.add_argument("--w_ghost", type=float, default=0.0,
                   help="灰印子项权重：惩罚掩码内输出的残留笔画状暗结构。默认 0=旧行为")
    p.add_argument("--w_hf", type=float, default=0.0,
                   help="高频细节项权重：带通信号的有符号 L1，保浅色细线/表格线。默认 0=旧行为")
    p.add_argument("--w_vgg", type=float, default=0.0,
                   help="VGG16 浅层(relu1_2/relu2_2)特征 L1 感知损失权重。"
                        "权重走缓存，不联网不新增依赖。默认 0=不启用")
    p.add_argument("--ghost_warmup", type=int, default=0,
                   help="前 N 步把 ghost 权重从 0 线性升到 --w_ghost，让占 71%% 损失的光照"
                        "归一先收敛，再逼模型彻底擦除。0=不预热（立即全权重）")
    p.add_argument("--mask_tau", type=float, default=0.12, help="局部对比度阈值：判定为笔迹")
    p.add_argument("--mask_white", type=float, default=0.60, help="GT 在该处的灰度下限：排除印刷内容")
    # 优化
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"])
    p.add_argument("--num_workers", type=int, default=4)
    # 记录
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--out_root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--price_per_hour", type=float, default=2.0, help="元/小时，用于费用估算")
    p.add_argument("--price_note", type=str, default="RTX 4090 租用参考价")
    p.add_argument("--replot", type=str, default="", help="只从该 run 目录的 log.csv 重绘曲线")
    return p.parse_args()


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / max(1, a.warmup)
    t = min(1.0, (step - a.warmup) / max(1, a.max_steps - a.warmup))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + np.cos(np.pi * t))


def replot(run_dir: Path, price: float) -> None:
    """从已有的 log.csv 重绘 curves.png（改绘图样式后不必重训）。"""
    cv = Curves(run_dir, price, "")
    with open(run_dir / "log.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cv.add({k: (v if k in ("step", "epoch") else float(v)) for k, v in row.items()})
    cv.plot(f"U-Net handwriting removal | {run_dir.name}")
    print(f"[重绘] {cv.png}  ({len(cv.rows)} 个评估点)")


def main():
    a = parse_args()
    if a.replot:
        replot(Path(a.replot), a.price_per_hour)
        return
    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None if a.amp == "off" or device.type != "cuda" else (
        torch.bfloat16 if a.amp == "bf16" else torch.float16)

    if not Path(a.splits).exists():
        print(f"[划分] {a.splits} 不存在，从 {a.data_root} 生成")
        build_splits(Path(a.data_root), Path(a.splits), a.downscale, limit=a.limit_data)
    splits = load_split(Path(a.splits))
    if not splits.get("train"):
        raise SystemExit("训练集为空")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{a.tag}" if a.tag else stamp
    out = Path(a.out_root) / name

    ep = a.eval_patch or a.patch
    tr_ds = PairedDataset(splits["train"], patch=a.patch, train=True, augment=DocAugment(),
                          downscale=a.downscale, limit=a.limit_train,
                          repeat=max(1, 512 // max(1, len(splits["train"]))))
    va_ds = PairedDataset(splits["val"] or splits["train"], patch=ep, train=False,
                          downscale=a.downscale, limit=a.eval_images)
    nw = a.num_workers if a.num_workers > 0 else 0
    tr_loader = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                           num_workers=nw, persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    # 验证集是确定性的中心裁剪，只解码一次常驻内存，省掉几十次重复解码
    items = [va_ds[i] for i in range(len(va_ds))]
    va_loader = DataLoader(_ListDS(items), batch_size=1)
    print(f"[数据] train={len(tr_ds)} (源 {len(splits['train'])} 对)  val={len(items)}  "
          f"patch={a.patch}/{ep}  downscale=1/{a.downscale}  workers={nw}")

    model = build_model(a.arch, 3, a.base_ch, a.depth, not a.no_residual, a.attn,
                        dim=a.dim, blocks=a.blocks, heads=a.heads,
                        rope=not a.no_rope).to(device)
    crit = RestorationLoss(a.w_l1, a.w_mse, a.w_ssim, a.w_edge, a.mask_weight, a.mask_tau,
                           mask_white=a.mask_white, w_ghost=a.w_ghost, w_hf=a.w_hf,
                           w_vgg=a.w_vgg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    start, best_p, best_s, best_ink, best_ink_step = 0, -1e9, -1e9, 1e9, 0
    if a.resume and Path(a.resume).exists():
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best_p, best_s = ck.get("step", 0), ck.get("best_psnr", -1e9), ck.get("best_ssim", -1e9)
        best_ink = ck.get("best_ink_err", 1e9)
        print(f"[恢复] {a.resume} step={start}")

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[模型] {a.arch} base={a.base_ch} depth={a.depth} residual={not a.no_residual} "
          f"attn={a.attn} | {model.n_params()/1e6:.3f} M 参数 | {gpu} | torch {torch.__version__}")
    (out).mkdir(parents=True, exist_ok=True)
    (out / "run_info.json").write_text(json.dumps(
        {"tag": a.tag, "args": vars(a), "gpu": gpu, "params_M": model.n_params() / 1e6,
         "torch": torch.__version__, "python": platform.python_version(),
         "started_at": stamp, "splits": {k: len(v) for k, v in splits.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    cv = Curves(out, a.price_per_hour, a.price_note)
    t0 = time.time()
    step, epoch, ema, done = start, 0, None, False
    step_t = time.time()
    print(f"[输出] {out}\n[开始] max_steps={a.max_steps} eval_every={a.eval_every} "
          f"price={a.price_per_hour} 元/小时 ({a.price_note})")

    while not done:
        epoch += 1
        for batch in tr_loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            lr = lr_at(step, a)
            for g in opt.param_groups:
                g["lr"] = lr

            crit.ghost_scale = (1.0 if a.ghost_warmup <= 0
                                else min(1.0, (step + 1) / a.ghost_warmup))
            with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                ld = crit(model(x), y, x)
                loss = ld["total"]
            if not torch.isfinite(loss):
                print(f"[警告] step {step} loss 非有限，跳过")
                opt.zero_grad(set_to_none=True)
                step += 1
                step_t = time.time()
                if step >= a.max_steps:
                    done = True
                    break
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()

            li = float(loss.detach())
            ema = li if ema is None else 0.98 * ema + 0.02 * li
            now = time.time()
            # 真实每步耗时（含数据加载），而不是只算模型那一段
            dt, step_t = now - step_t, now
            step += 1

            if step % a.print_every == 0 or step == 1:
                el = (now - t0) / 60
                print(f"  step {step:5d}/{a.max_steps} | loss {ema:.4f} | lr {lr:.2e} "
                      f"| {dt:.2f} s/step | {el:.1f} min | {el/60*a.price_per_hour:.2f} 元")

            if step % a.eval_every == 0 or step >= a.max_steps:
                m = evaluate(model, va_loader, crit, device, dtype)
                el_h = (now - t0) / 3600
                cv.add({"step": step, "epoch": epoch, "lr": lr, "loss": float(ema),
                        "val_loss": m["val_loss"], "l1": float(ld["l1"]), "mse": float(ld["mse"]),
                        "ssim_loss": float(ld["ssim_loss"]), "edge": float(ld["edge"]),
                        "ghost_loss": float(ld["ghost"]), "hf_loss": float(ld["hf"]),
                        "vgg_loss": float(ld["vgg"]),
                        "psnr": m["psnr"], "ssim": m["ssim"], "sec_per_step": dt,
                        "elapsed_min": (now - t0) / 60, "cost_yuan": el_h * a.price_per_hour,
                        **{k: m[k] for k in INK_KEYS}})
                cv.flush()
                # 曲线数据每 eval_every 步记录一次（验收要求），
                # 但 PNG 不必每次都重绘——20000 步会有 1000 次评估，绘图本身很费时间
                if step % max(a.eval_every, 100) == 0 or step >= a.max_steps:
                    cv.plot(f"U-Net handwriting removal | {name}")

                flag = ""
                if m["psnr"] > best_p:
                    best_p = m["psnr"]
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                "args": vars(a), "best_psnr": best_p, "best_ssim": best_s},
                               out / "best.pth")
                    flag = "  <- best PSNR"
                best_s = max(best_s, m["ssim"])
                # 另存一份"手写区误差最小"的权重：PSNR 由光度归一化主导，
                # 两者最优的 step 未必是同一个，推理时两个都试。
                if m["ink_err"] == m["ink_err"] and m["ink_err"] < best_ink:
                    best_ink, best_ink_step = m["ink_err"], step
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                "args": vars(a), "best_psnr": best_p, "best_ssim": best_s,
                                "best_ink_err": best_ink},
                               out / "best_ink.pth")
                    flag += f"  <- best ink_err {best_ink:.4f}"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "args": vars(a), "best_psnr": best_p, "best_ssim": best_s},
                           out / "last.pth")
                (out / "cost.json").write_text(json.dumps({
                    "total_seconds": now - t0, "total_hours": el_h, "steps": step,
                    "sec_per_step": dt, "price_per_hour": a.price_per_hour, "price_note": a.price_note,
                    "estimated_cost_yuan": el_h * a.price_per_hour, "gpu": gpu,
                    "params_M": model.n_params() / 1e6}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                print(f"  [eval] step {step:5d} | PSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.4f} "
                      f"| 手写区 {m['ink_err']:.4f} | 非手写区 {m['bg_err']:.4f} "
                      f"| 灰印子 {m['ghost']:.4f} ({m['ghost_keep']*100:.1f}%) "
                      f"| {el_h*a.price_per_hour:.2f} 元{flag}")

            if step >= a.max_steps:
                done = True
                break

    h = (time.time() - t0) / 3600
    print("=" * 68)
    print(f"[完成] {step} steps | {h*60:.1f} 分钟 ({h:.3f} 小时)")
    print(f"[费用] {a.price_per_hour} 元/小时 x {h:.3f} h = {h*a.price_per_hour:.2f} 元")
    print(f"[指标] best PSNR {best_p:.2f} dB | best SSIM {best_s:.4f}")
    print(f"[指标] best 手写区误差 {best_ink:.4f} (step {best_ink_step})  <- 验收看这个")
    print(f"[产物] {out}/log.csv  {out}/curves.png  {out}/best.pth  {out}/best_ink.pth")


if __name__ == "__main__":
    main()
