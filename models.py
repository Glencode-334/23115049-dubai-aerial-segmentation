"""Tasks 2/3 - Baseline U-Net and the single controlled change (dilated bottleneck).

Only `self.bottleneck` differs between the two models; encoder, decoder, skips and head are
built by identical code. Run `python models.py` for the block-by-block parameter table.
"""
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        pad = dilation * (k - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(ConvBNReLU(in_ch, out_ch), ConvBNReLU(out_ch, out_ch))


class ParallelDilatedBottleneck(nn.Module):
    """Stem 3x3 256->512, 4 parallel 3x3 branches 512->128 (dilation 1,2,4,8), concat, 1x1 512->512."""

    def __init__(self, in_ch, out_ch, rates=(1, 2, 4, 8)):
        super().__init__()
        self.rates = rates
        self.stem = ConvBNReLU(in_ch, out_ch)
        self.branches = nn.ModuleList([ConvBNReLU(out_ch, out_ch // len(rates), dilation=r) for r in rates])
        self.project = ConvBNReLU(out_ch, out_ch, k=1)

    def forward(self, x):
        x = self.stem(x)
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class SegmentationUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base_ch=32, use_dilated_bottleneck=False):
        super().__init__()
        c = [base_ch * 2 ** i for i in range(5)]  # 32,64,128,256,512
        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        # ---- the ONLY difference between the two models ----
        self.bottleneck = ParallelDilatedBottleneck(c[3], c[4]) if use_dilated_bottleneck else DoubleConv(c[3], c[4])
        # -----------------------------------------------------
        self.up4 = UpBlock(c[4], c[3])
        self.up3 = UpBlock(c[3], c[2])
        self.up2 = UpBlock(c[2], c[1])
        self.up1 = UpBlock(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d = self.up4(b, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        return self.head(d)


@torch.no_grad()
def count_conv_macs(model, size=256):
    """Multiply-accumulates of all Conv2d/ConvTranspose2d layers for one 1x3xHxW forward pass
    (BN/ReLU/pool/concat ignored). Hardware-independent complement to measured latency."""
    total = [0]

    def hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        if isinstance(m, nn.ConvTranspose2d):  # each input pixel scatters to k*out_ch outputs
            total[0] += inp[0].numel() // inp[0].shape[0] * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
        else:
            total[0] += out.numel() // out.shape[0] * k

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    was_training = model.training
    model.eval()
    model(torch.zeros(1, 3, size, size, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hs:
        h.remove()
    return total[0]


BLOCKS = ["enc1", "enc2", "enc3", "enc4", "bottleneck", "up4", "up3", "up2", "up1", "head"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def block_param_table(base, var):
    rows = []
    for b in BLOCKS:
        pb, pv = count_parameters(getattr(base, b)), count_parameters(getattr(var, b))
        rows.append((b, pb, pv, pv - pb))
    rows.append(("TOTAL", count_parameters(base), count_parameters(var),
                 count_parameters(var) - count_parameters(base)))
    return rows


def format_param_table(rows):
    lines = [f"{'block':<12}{'baseline':>12}{'dilated':>12}{'delta':>12}", "-" * 48]
    for b, pb, pv, d in rows:
        if b == "TOTAL":
            lines.append("-" * 48)
        lines.append(f"{b:<12}{pb:>12,}{pv:>12,}{d:>+12,}")
    return "\n".join(lines)


if __name__ == "__main__":
    base = SegmentationUNet(use_dilated_bottleneck=False)
    var = SegmentationUNet(use_dilated_bottleneck=True)
    print(format_param_table(block_param_table(base, var)))
    x = torch.randn(2, 3, 256, 256)
    print("output shapes:", tuple(base(x).shape), tuple(var(x).shape))
    mb, mv = count_conv_macs(base), count_conv_macs(var)
    print(f"conv MACs @256x256: baseline {mb / 1e9:.3f} G, dilated {mv / 1e9:.3f} G, delta {(mv - mb) / 1e6:+.1f} M")
