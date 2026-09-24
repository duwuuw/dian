"""模型 / 损失 / 指标。

U-Net 采用**残差学习**：网络预测"需要擦掉的内容"，输出 out = clamp(x - pred)。
擦除任务里被改动的像素只占少数，学残差比从零重建整页更快、印刷体保真度更高。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================ U-Net
def _norm(c):
    return nn.GroupNorm(min(8, c), c)      # 小 batch 下比 BatchNorm 稳，且推理行为一致


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _norm(cout), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _norm(cout), nn.LeakyReLU(0.1, True))

    def forward(self, x):
        return self.b(x)


class _Attn(nn.Module):
    """Attention Gate：用解码器特征门控 skip 特征，抑制手写区域向解码器传播。"""

    def __init__(self, cskip, cgate):
        super().__init__()
        m = max(1, cskip // 2)
        self.wx, self.wg = nn.Conv2d(cskip, m, 1), nn.Conv2d(cgate, m, 1)
        self.psi = nn.Conv2d(m, 1, 1)

    def forward(self, x, g):
        return x * torch.sigmoid(self.psi(F.leaky_relu(self.wx(x) + self.wg(g), 0.1)))


class UNet(nn.Module):
    def __init__(self, ch=3, base=32, depth=4, residual=True, attn=False):
        super().__init__()
        self.depth, self.residual = depth, residual
        cs = [base * 2 ** i for i in range(depth + 1)]
        self.stem = _DoubleConv(ch, cs[0])
        self.down = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), _DoubleConv(cs[i], cs[i + 1])) for i in range(depth)])
        self.attn = nn.ModuleList([
            _Attn(cs[i], cs[i + 1]) if attn else None for i in range(depth - 1, -1, -1)])
        self.up = nn.ModuleList([_DoubleConv(cs[i + 1] + cs[i], cs[i]) for i in range(depth - 1, -1, -1)])
        self.head = nn.Conv2d(cs[0], ch, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        H, W = (h + m - 1) // m * m, (w + m - 1) // m * m
        xp = F.pad(x, (0, W - w, 0, H - h), mode="replicate") if (H != h or W != w) else x
        feats = [self.stem(xp)]
        for d in self.down:
            feats.append(d(feats[-1]))
        y = feats[-1]
        for i, u in enumerate(self.up):
            s = feats[-2 - i]
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            if self.attn[i] is not None:
                s = self.attn[i](s, y)
            y = u(torch.cat([s, y], 1))
        out = self.head(y)[..., :h, :w]
        return x - out if self.residual else torch.sigmoid(out)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================ DiT 风格 Transformer
def rope2d(h, w, hd, device, theta=10000.0, dtype=torch.float32):
    """2D RoPE 的 cos/sin 表（Next-DiT / FiT 的通行做法）。

    为什么必须有：原来用可学习位置编码表 pe_grid=64，训练 patch 256 时 token 网格
    正好 64x64 对得上，但推理 tile=512 时网格是 128x128，只能 F.interpolate 插值
    —— 训练与推理的位置编码口径不一致，白白损失精度。RoPE 没有位置参数表，
    token 位置直接由坐标决定，换分辨率天然一致。

    做法：把 head_dim 均分成两半，前半按行坐标(i)旋转、后半按列坐标(j)旋转，
    每半内部再两两成对当作复数乘 e^{iθ}。位置信息只进 q/k，不进参数。
    """
    d = hd // 4                      # 每轴 hd//2 维、每 2 维一组 -> hd//4 个频率
    assert d >= 1, f"head_dim={hd} 太小，2D RoPE 至少要 4"
    inv = theta ** (-torch.arange(0, d, device=device, dtype=dtype) / d)
    ay = torch.outer(torch.arange(h, device=device, dtype=dtype), inv)      # (h, d)
    ax = torch.outer(torch.arange(w, device=device, dtype=dtype), inv)      # (w, d)
    ay = ay[:, None].expand(h, w, d).reshape(1, 1, h * w, d)
    ax = ax[None, :].expand(h, w, d).reshape(1, 1, h * w, d)
    return ay.cos(), ay.sin(), ax.cos(), ax.sin()


def _rotate(t, cos, sin):
    """把 t 的最后一维两两成对当作复数，乘 e^{iθ}。"""
    v = t.view(*t.shape[:-1], -1, 2)
    a, b = v[..., 0], v[..., 1]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


def _apply_rope(t, r):
    """t: (B, heads, N, hd)；r: rope2d 的四个表。前半用行角、后半用列角。"""
    cy, sy, cx, sx = (x.to(t.dtype) for x in r)
    hd = t.shape[-1]
    return torch.cat([_rotate(t[..., :hd // 2], cy, sy),
                      _rotate(t[..., hd // 2:], cx, sx)], -1)


class _Block(nn.Module):
    """pre-norm Transformer block。用 F.scaled_dot_product_attention，
    走 flash/efficient 内核，注意力矩阵不落地，显存是 O(N) 而不是 O(N²)。"""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x, rope=None):
        B, N, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            q, k = _apply_rope(q, rope), _apply_rope(k, rope)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(o)
        return x + self.mlp(self.n2(x))


class DiTRestore(nn.Module):
    """DiT 风格的 Transformer 修复模型（**直接回归**，不做扩散采样）。

    为什么不是真·DiT：从零训练一个扩散模型在 30 元预算内不可能（小图也要
    几百 GPU·小时）。这里保留 DiT 的核心结构（patch 化 token + 全局自注意力 +
    可学习位置编码），但把输出头改成直接预测残差，因此单步就能出结果，
    训练成本与 U-Net 同量级，可以直接对比。

    为什么保留全分辨率 CNN 分支：纯 patch-level 的 ViT 在 /4 分辨率上做注意力，
    手写细笔画和表格线会先被降采样丢掉，这类任务上反而比 U-Net 差。
    所以这里用「CNN stem 保细节 + Transformer 瓶颈管全局版式」的混合结构，
    这也是 NAFNet/Restormer 一类修复网络的通行做法。
    """

    def __init__(self, ch=3, base=32, dim=256, blocks=6, heads=8, residual=True,
                 pe_grid=64, rope=True):
        super().__init__()
        self.residual, self.rope = residual, rope
        self.stem = _DoubleConv(ch, base)                                   # /1
        self.d1 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(base, base * 2))   # /2
        self.d2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base * 2, dim, 1))   # /4
        if not rope:                        # 消融用：旧的"可学习位置编码 + 推理时插值"
            assert dim % heads == 0 and (dim // heads) % 4 == 0
            self.pe = nn.Parameter(torch.zeros(1, dim, pe_grid, pe_grid))
            nn.init.trunc_normal_(self.pe, std=0.02)
        self._rcache = {}
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(blocks)])
        self.norm = nn.LayerNorm(dim)
        self.u1 = _DoubleConv(dim + base * 2, base * 2)                     # /2
        self.u2 = _DoubleConv(base * 2 + base, base)                        # /1
        self.head = nn.Conv2d(base, ch, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        m = 4
        H, W = (h + m - 1) // m * m, (w + m - 1) // m * m
        xp = F.pad(x, (0, W - w, 0, H - h), mode="replicate") if (H != h or W != w) else x

        s1 = self.stem(xp)                      # /1
        s2 = self.d1(s1)                        # /2
        z = self.d2(s2)                         # /4, dim
        B, C, hz, wz = z.shape

        if self.rope:
            key = (hz, wz, z.device)
            r = self._rcache.get(key)
            if r is None:                        # 按 (h,w) 缓存，训练时尺寸固定只算一次
                r = rope2d(hz, wz, z.shape[1] // self.blocks[0].h, z.device)
                if len(self._rcache) > 8:        # 推理 tile 尺寸变化时别无限增长
                    self._rcache.clear()
                self._rcache[key] = r
            t = z.flatten(2).transpose(1, 2)
            for blk in self.blocks:
                t = blk(t, r)
        else:
            pe = self.pe
            if pe.shape[-2:] != (hz, wz):        # 推理 tile 尺寸变化时插值位置编码
                pe = F.interpolate(pe, size=(hz, wz), mode="bilinear", align_corners=False)
            t = (z + pe).flatten(2).transpose(1, 2)
            for blk in self.blocks:
                t = blk(t)
        z = self.norm(t).transpose(1, 2).reshape(B, C, hz, wz)

        y = self.u1(torch.cat([s2, F.interpolate(z, size=s2.shape[-2:], mode="bilinear",
                                                 align_corners=False)], 1))
        y = self.u2(torch.cat([s1, F.interpolate(y, size=s1.shape[-2:], mode="bilinear",
                                                 align_corners=False)], 1))
        out = self.head(y)[..., :h, :w]
        return x - out if self.residual else torch.sigmoid(out)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(arch, ch, base, depth, residual, attn, **kw):
    if arch == "dit":
        return DiTRestore(ch, base, kw.get("dim", 256), kw.get("blocks", 6), kw.get("heads", 8),
                          residual, rope=kw.get("rope", True))
    return UNet(ch, base, depth, residual, attn)


# ================================================================ SSIM / PSNR
def _gauss_win(ws, sigma, ch, device, dtype):
    c = torch.arange(ws, dtype=dtype, device=device) - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return ((g.t() @ g).view(1, 1, ws, ws)).expand(ch, 1, ws, ws).contiguous()


def _gauss_blur(t, sigma, ksize=None):
    """高斯低通。用于"高频带"的定义：hi(t) = t - blur(t)。"""
    ks = ksize or (int(2 * round(3 * sigma)) + 1)
    win = _gauss_win(ks, sigma, t.shape[1], t.device, t.dtype)
    return F.conv2d(t, win, padding=ks // 2, groups=t.shape[1])


def _local_darkness(t, ksize=31):
    """t 比它的邻域均值暗多少（relu 后 >0 即"细笔画状结构"）。

    与 proxy_mask 同一判据：阴影是低频的（bg−g≈0），笔迹是高频的，
    所以这个量只对笔画有响应，对光照梯度没有。avg_pool2d 边界零填充会让
    bg 偏暗、bg−g 偏负，relu 后归零 —— 边界不会产生假信号，方向是安全的。
    """
    g = t.mean(1, keepdim=True)
    return torch.clamp(F.avg_pool2d(g, ksize, stride=1, padding=ksize // 2) - g, min=0.0)


def ssim(x, y, ws=11, sigma=1.5, data_range=1.0, mean=True, crop=True):
    """与 skimage.metrics.structural_similarity(gaussian_weights=True) 逐位对齐。

    crop=True 会裁掉 SSIM map 边缘 ws//2 像素再平均（skimage 的默认行为），
    这样训练曲线与推理评测口径完全一致。

    强制 fp32：文档图是大面积近白底，局部方差极小，SSIM 靠 E[x²]−μ² 相减得到，
    bf16 的 8 位尾数在这种相减里会完全失效（实测同一输入的 1-SSIM 从 0.21 变成 0.44，
    还会出现负值）。所以这里显式关掉 autocast，全程用 float32 计算。
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        ch, pad = x.shape[1], ws // 2
        win = _gauss_win(ws, sigma, ch, x.device, x.dtype)
        k = lambda t: F.conv2d(t, win, padding=pad, groups=ch)
        mx, my = k(x), k(y)
        sxx, syy, sxy = k(x * x) - mx * mx, k(y * y) - my * my, k(x * y) - mx * my
        c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        if crop:
            m = m[..., pad:-pad, pad:-pad]
        return m.mean() if mean else m.mean(dim=(1, 2, 3))


@torch.no_grad()
def psnr(x, y, data_range=1.0):
    return 10 * torch.log10(data_range ** 2 / (x - y).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12))


# ================================================================ 损失
def proxy_mask(inp, target, tau=0.12, white=0.60, ksize=31):
    """手写代理掩码（无像素级标注时用）。

    不能用 |输入 − GT| 直接当掩码：本数据集的输入是拍摄图，大面积阴影造成的
    灰度差异比笔迹还大，那样标出来的是"阴影区"而不是"手写区"，加权等于没加。

    这里的判据是**局部对比度**：
        输入中比它的邻域均值明显更暗（= 细笔画），且 GT 在该处是白底（= 没有印刷内容）。
    阴影是低频的（邻域均值和自身接近，bg − g ≈ 0），笔迹是高频的，两者可以分开。
    """
    g_in = inp.mean(1, keepdim=True)
    bg = F.avg_pool2d(g_in, ksize, stride=1, padding=ksize // 2)
    g_t = target.mean(1, keepdim=True)
    return ((bg - g_in > tau) & (g_t > white)).float()


@torch.no_grad()
def ink_metrics(pred, target, inp, tau=0.12, white=0.60, ksize=31) -> dict:
    """手写区误差分解（AGENT.md 5.2）——训练中每步评估都算，不能只看 PSNR。

    本任务里光度归一化贡献约 71% 的损失，手写擦除只占 29%，PSNR 会被前者主导。
    实测第一版模型 PSNR 涨得很漂亮但手写根本没擦掉，所以这两个数必须单独看：
        ink_err  手写区 |pred-GT|    —— 擦得干不干净
        bg_err   非手写区 |pred-GT|  —— 有没有为了擦手写而误伤印刷内容
        ghost    掩码内残留的笔画状暗结构 —— 灰印子的直接度量（基线 0.095）
        ghost_keep = ghost / 输入侧同类量，即"原始笔画能量还剩百分之几"（基线 37%）
    """
    p, t, i = pred.float(), target.float(), inp.float()
    m = proxy_mask(i, t, tau, white, ksize).bool()
    lo = m.flatten(1).sum(1) >= 50                       # 掩码太小的样本不计入
    if not bool(lo.any()):
        return {}
    d = (p - t).abs().mean(1)
    dark_p = _local_darkness(p, ksize)[:, 0]
    dark_i = _local_darkness(i, ksize)[:, 0]
    sel = lambda x, mm: x.flatten(1)[mm.flatten(1)].mean()
    g_out = sel(dark_p, m)
    g_in = sel(dark_i, m).clamp_min(1e-6)
    return {"ink_err": float(sel(d, m)), "bg_err": float(sel(d, ~m)),
            "ghost": float(g_out), "ghost_keep": float(g_out / g_in)}


class PerceptualLoss(nn.Module):
    """VGG16 中间层特征的 L1 感知损失。

    权重是 torchvision 的 ImageNet 预训练 vgg16，已缓存在
    ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth，**不需要联网、不新增依赖**
    （没装 lpips，但 VGG perceptual 用 torchvision 自带实现即可）。

    只取浅层（默认 relu1_2 / relu2_2）：本任务的失效模式是"灰印子"和"细线被冲淡"，
    都是纹理/边缘层面的事，正是浅层特征敏感的东西；深层语义特征对"白纸上有没有
    淡灰色笔迹"这种差异反而不敏感，还更贵。

    注意：全程 fp32（外层 RestorationLoss 已关 autocast）。这个项目已经在 bf16
    精度上栽过一次（见 ssim 的注释），感知特征不再冒这个险。
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, layers=(3, 8)):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        net = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.layers = list(layers)
        self.net = net[:max(self.layers) + 1].eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self._MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self._STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, pred, target):
        x = (pred.float() - self.mean) / self.std
        y = (target.float() - self.mean) / self.std
        loss, n = pred.new_zeros(()), 0
        for i, m in enumerate(self.net):
            x, y = m(x), m(y)
            if i in self.layers:
                loss, n = loss + F.l1_loss(x, y), n + 1
        return loss / max(n, 1)


class RestorationLoss(nn.Module):
    """L = w_l1·L1 + w_mse·MSE + w_ssim·(1-SSIM) + w_edge·Edge + w_ghost·Ghost + w_hf·HF

    为什么要组合：
      MSE   对大误差敏感、优化稳，但会产生"平均化"的模糊结果，细笔画被抹平；
      L1    对离群点鲁棒、边缘锐，但梯度恒定，平坦区后期容易震荡；
      SSIM  直接优化结构相似度，保住笔画结构；
      Edge  Sobel 梯度域 L1，专门保表格线和字符轮廓，抑制"擦不干净"和"擦出空洞"。

    mask_weight>0 时用 |输入-GT| 差异图膨胀后作为"手写区域"的代理掩码给 L1 加权
    （本任务没有像素级标注），让损失更关注需要擦除的地方。

    针对两个已知失效模式补的两项（默认 0，不改变原行为）：
      Ghost  L1/L2 回归的解是条件中位数/均值：模型没把握时"少擦一点"最安全，
             于是手写被减掉 ~75% 后留下**淡灰印子**。实测基线在掩码区还残留
             37% 的笔画暗结构（输入 0.255 -> 输出 0.095），肉眼可辨。
             光靠像素 L1 治不了——它在掩码区给的梯度被整片区域摊薄了。
             Ghost 项直接把"输出在掩码内还比邻域暗多少"当损失，是针对症的探针。
      HF     浅色细线被光照归一化冲淡：残差里混进了高频成分，把印刷线/插画轮廓
             一起减掉了。Edge(Sobel) 只比梯度**幅值**、对相位不敏感，线整体变浅
             也照样匹配；HF 比的是带通信号的**有符号**差值，线浅了、移位了都罚。
    """

    _SOBEL_X = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
    _SOBEL_Y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])

    def __init__(self, w_l1=1.0, w_mse=0.0, w_ssim=0.2, w_edge=0.2,
                 mask_weight=0.0, mask_tau=0.12, mask_dilate=7, mask_white=0.60,
                 w_ghost=0.0, w_hf=0.0, ghost_ksize=31, hf_sigma=3.0, w_vgg=0.0):
        super().__init__()
        self.w = dict(l1=w_l1, mse=w_mse, ssim=w_ssim, edge=w_edge,
                      ghost=w_ghost, hf=w_hf, vgg=w_vgg)
        # w_vgg=0 时不构造 VGG（省掉 528MB 权重加载的十几秒启动开销）
        self.vgg = PerceptualLoss() if w_vgg > 0 else None
        self.mask_weight, self.mask_tau, self.mask_dilate = mask_weight, mask_tau, mask_dilate
        self.mask_white = mask_white
        self.ghost_ksize, self.hf_sigma = ghost_ksize, hf_sigma
        self.ghost_scale = 1.0      # 训练时由 train.py 按 warmup 进度调低，让光度归一先收敛
        self.register_buffer("sx", self._SOBEL_X.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sy", self._SOBEL_Y.view(1, 1, 3, 3), persistent=False)

    def _grad(self, t):
        ch = t.shape[1]
        gx = F.conv2d(t, self.sx.to(t.dtype).expand(ch, 1, 3, 3), padding=1, groups=ch)
        gy = F.conv2d(t, self.sy.to(t.dtype).expand(ch, 1, 3, 3), padding=1, groups=ch)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target, inp=None):
        # 整个损失在 fp32 下算：SSIM 的方差是 E[x²]−μ² 相减得到，
        # 文档图近白底、局部方差极小，bf16 会直接算错；Edge 里的 sqrt(·+1e-6) 同理。
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            inp = inp.float() if inp is not None else None

            wmap, m_ink = None, None
            if (self.mask_weight > 0 or self.w["ghost"] > 0) and inp is not None:
                m_ink = proxy_mask(inp, target, self.mask_tau, self.mask_white)
                if self.mask_weight > 0:
                    m = m_ink
                    if self.mask_dilate > 1:
                        k = self.mask_dilate | 1
                        m = F.max_pool2d(m, k, stride=1, padding=k // 2)
                    wmap = 1.0 + self.mask_weight * m

            diff = (pred - target).abs()
            l1 = diff.mean() if wmap is None else (diff * wmap).sum() / wmap.sum().clamp_min(1e-6)
            mse = (pred - target).pow(2).mean()
            sl = 1.0 - ssim(pred, target)
            el = (self._grad(pred) - self._grad(target)).abs().mean()

            # 灰印子：掩码内输出了多少"比邻域暗"的笔画状结构。目标为 0（该处 GT 是白底）。
            ghost = pred.new_zeros(())
            if self.w["ghost"] > 0 and m_ink is not None and float(m_ink.sum()) > 0:
                # 取 |暗度差| 而不是 relu(暗度)：掩码区的 GT 是平坦白纸，局部暗度应当≈0。
                # 只罚"还暗着"会留漏洞 —— 把掩码区整体提亮成光晕，relu 那一侧同样是 0，
                # 而实测那样会把非手写区误差推高 49%。取绝对值把"过亮"一并罚掉。
                ghost = ((_local_darkness(pred, self.ghost_ksize)
                          - _local_darkness(target, self.ghost_ksize)).abs() * m_ink
                         ).sum() / m_ink.sum().clamp_min(1e-6)

            # 高频细节：带通信号的有符号 L1，保浅色细线/表格线/插画轮廓。
            hf = pred.new_zeros(())
            if self.w["hf"] > 0:
                hi = lambda t: t.mean(1, keepdim=True) - _gauss_blur(t.mean(1, keepdim=True), self.hf_sigma)
                hf = (hi(pred) - hi(target)).abs().mean()

            vgg = pred.new_zeros(())
            if self.w["vgg"] > 0 and self.vgg is not None:
                vgg = self.vgg(pred, target)

            total = (self.w["l1"] * l1 + self.w["mse"] * mse + self.w["ssim"] * sl
                     + self.w["edge"] * el
                     + self.w["ghost"] * self.ghost_scale * ghost + self.w["hf"] * hf
                     + self.w["vgg"] * vgg)
        parts = dict(l1=l1.detach(), mse=mse.detach(), ssim_loss=sl.detach(),
                     edge=el.detach(), ghost=ghost.detach(), hf=hf.detach(), vgg=vgg.detach())
        return {"total": total, **parts}


# ================================================================ 坏结果检测
def blank_report(img: np.ndarray, name="", gt=None, inp=None):
    """检测退化输出。

    注意：干净试卷本身就是大面积白底（实测 mean≈0.93），所以"白色像素多"绝不能判异常。
    这里抓的是真正的退化：全白 / 全黑 / 对比度丢失 / 墨迹保留率过低。
    """
    x = np.asarray(img, np.float32)
    m, s = float(x.mean()), float(x.std())
    ink = float((x.mean(2 if x.ndim == 3 else 0) < 0.6).mean())
    flags = []
    if m > 0.985 and s < 0.02:
        flags.append("BLANK_WHITE")
    if m < 0.03 and s < 0.02:
        flags.append("BLANK_BLACK")
    if s < 0.02:
        flags.append("LOW_CONTRAST")

    rec = {"name": name, "mean": m, "std": s, "ink_out": ink}
    ref = gt if gt is not None else inp
    if ref is not None:
        r = np.asarray(ref, np.float32)
        ink_ref = float((r.mean(2 if r.ndim == 3 else 0) < 0.6).mean())
        keep = ink / max(ink_ref, 1e-6)
        rec["ink_keep"] = keep
        thr = 0.5 if gt is not None else 0.15
        if ink_ref > 0.002 and keep < thr:
            flags.append("INK_LOST")
        if gt is not None and inp is not None:
            d_in = float(np.abs(np.asarray(inp, np.float32) - r).mean())
            d_out = float(np.abs(x - r).mean())
            if d_in > 1e-6:
                rec["removed_frac"] = 1.0 - d_out / d_in   # 输入与GT的差异被消除了多少
    rec["ok"] = not flags
    rec["flags"] = flags
    return rec


def psnr_np(pred, target, data_range=1.0) -> float:
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(data_range ** 2 / mse))


def ssim_np(pred, target, data_range=1.0) -> float:
    from skimage.metrics import structural_similarity as f
    kw = dict(data_range=data_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return float(f(pred, target, **kw) if pred.ndim == 2 else f(pred, target, channel_axis=2, **kw))


if __name__ == "__main__":
    import numpy as np
    import sys
    a = torch.rand(2, 3, 128, 128)
    b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
    from skimage.metrics import structural_similarity as sk
    v1 = ssim(a, b).item()
    v2 = float(np.mean([sk(a[i].permute(1, 2, 0).numpy(), b[i].permute(1, 2, 0).numpy(),
                           channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5,
                           use_sample_covariance=False) for i in range(2)]))
    print(f"SSIM torch={v1:.6f} skimage={v2:.6f} diff={abs(v1-v2):.2e}")
    assert abs(v1 - v2) < 1e-4, "SSIM 与 skimage 不一致"

    net = UNet(base=16)
    for shp in [(2, 3, 256, 256), (1, 3, 251, 373)]:
        assert net(torch.randn(*shp)).shape == shp
    print(f"UNet 前向 OK, 参数 {net.n_params()/1e6:.3f} M")

    # 文档图场景（大面积近白底、方差极小）下 SSIM 的数值稳定性
    if torch.cuda.is_available():
        torch.manual_seed(0)
        t = (torch.rand(4, 3, 256, 256) * 0.05 + 0.93).clamp(0, 1)
        t[:, :, 50:80, 50:200] = 0.1
        pr = (t + 0.02 * torch.randn_like(t)).clamp(0, 1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v = ssim(pr.cuda(), t.cuda()).item()
        ref = ssim(pr, t).item()
        print(f"autocast 下 SSIM={v:.6f}  fp32 参考={ref:.6f}  diff={abs(v-ref):.2e}")
        assert abs(v - ref) < 1e-4 and v <= 1.0 + 1e-6, "SSIM 在 autocast 下数值失效"

    p = torch.rand(1, 3, 64, 64, requires_grad=True)
    l = RestorationLoss(mask_weight=1.0)(p, torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    l["total"].backward()
    assert torch.isfinite(p.grad).all()
    assert float(l["ssim_loss"]) >= -1e-6, f"ssim_loss 出现负值: {float(l['ssim_loss'])}"
    print("[OK] 组合损失与反向传播自检通过")

    # 新增两项的自检：默认权重必须完全复现旧行为；开启后梯度有限、量纲正确
    torch.manual_seed(0)
    t = (torch.rand(1, 3, 96, 96) * 0.1 + 0.88).clamp(0, 1)
    i = t.clone()                                                   # 3px 细笔画才符合
    i[:, :, 20:76, 42:45] = 0.05                                    # 局部对比度掩码的判据
    i[:, :, 20:76, 60:63] = 0.05
    base = RestorationLoss(mask_weight=5.0)(i, t, i)
    assert float(base["ghost"]) == 0.0 and float(base["hf"]) == 0.0, "默认权重不应改变旧行为"
    full = RestorationLoss(mask_weight=5.0, w_ghost=1.0, w_hf=1.0)(i, t, i)
    assert float(full["ghost"]) > 0.05, f"未擦除时 ghost 应显著>0，实际 {float(full['ghost'])}"
    # 干净输出只剩纸面噪声引起的起伏（~1e-2），比未擦除低一个数量级以上
    clean = RestorationLoss(mask_weight=5.0, w_ghost=1.0, w_hf=1.0)(t.clone(), t, i)
    assert float(clean["ghost"]) < 1e-6, \
        f"输出==GT 时 ghost 必须恰好为 0，实际 {float(clean['ghost'])}"
    # 对称性自检：把掩码区整体提亮成"光晕"（relu 版本会漏掉这个退化解）必须被罚
    halo = t.clone(); halo[:, :, 20:76, 42:63] = 1.0
    hl = RestorationLoss(mask_weight=5.0, w_ghost=1.0)(halo, t, i)
    assert float(hl["ghost"]) > 1e-4, "光晕退化解没有被罚，ghost 项仍有漏洞"
    q = t.clone().requires_grad_(True)
    RestorationLoss(mask_weight=5.0, w_ghost=1.0, w_hf=1.0)(q, t, i)["total"].backward()
    assert torch.isfinite(q.grad).all() and float(q.grad.abs().sum()) > 0
    # DiT + RoPE：换分辨率必须照样跑通（这正是 RoPE 要解决的问题）
    for shp in [(2, 3, 256, 256), (1, 3, 512, 512), (1, 3, 251, 373)]:
        d = DiTRestore(base=16, dim=64, blocks=2, heads=8)
        assert d(torch.randn(*shp)).shape == shp, f"DiT+RoPE 前向失败 {shp}"
    d = DiTRestore(base=16, dim=64, blocks=2, heads=8, rope=False)
    assert d(torch.randn(1, 3, 256, 256)).shape == (1, 3, 256, 256)
    assert "pe" not in dict(DiTRestore(base=16, dim=64, blocks=2).named_parameters()), \
        "RoPE 模式下不应再有位置编码参数表"
    # RoPE 的核心性质：旋转角只由坐标决定。4x4 网格的 token#4 与 8x8 网格的 token#8
    # 都是 (row=1, col=0)，必须得到完全相同的角度 —— 这就是换分辨率不用插值的根据。
    r4, r8 = rope2d(4, 4, 32, "cpu"), rope2d(8, 8, 32, "cpu")
    for k, nm in [(0, "cos_y"), (2, "cos_x")]:
        assert torch.allclose(r4[k][0, 0, 4], r8[k][0, 0, 8], atol=1e-6), f"{nm} 与网格尺寸相关了"
    print("[OK] DiT + 2D RoPE 自检通过（分辨率无关，无位置参数表）")

    mu = ink_metrics(i, t, i)                                        # 完全没擦：残留应接近 100%
    me = ink_metrics(t, t, i)                                        # 擦干净：残留应接近 0
    assert mu["ghost_keep"] > 0.9, f"未擦除时 ghost_keep 应接近 1，实际 {mu}"
    assert me["ghost_keep"] < 0.1 and me["ink_err"] < mu["ink_err"] * 0.1, \
        f"ink_metrics 自检失败: 未擦 {mu} / 已擦 {me}"
    print(f"[OK] ghost/HF 项自检通过 (ghost={float(full['ghost']):.4f} 完美输出={float(clean['ghost']):.2e})")
