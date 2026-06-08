# Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior" (arXiv:2008.13751)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


def down(c_in, c_out):
    return nn.Conv2d(c_in, c_out, 2, stride=2, bias=False)


def up(c_in, c_out):
    return nn.ConvTranspose2d(c_in, c_out, 2, stride=2, bias=False)


class DRUNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, base=32, n_blocks=2, residual=True, gradient_checkpointing=True):
        super().__init__()
        self.residual = residual
        self.gradient_checkpointing = gradient_checkpointing
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.df = 8

        self.head = nn.Conv2d(in_ch, c1, 3, padding=1, bias=False)
        self.enc1 = nn.Sequential(*[ResBlock(c1) for _ in range(n_blocks)])
        self.down1 = down(c1, c2)
        self.enc2 = nn.Sequential(*[ResBlock(c2) for _ in range(n_blocks)])
        self.down2 = down(c2, c3)
        self.enc3 = nn.Sequential(*[ResBlock(c3) for _ in range(n_blocks)])
        self.down3 = down(c3, c4)
        self.body = nn.Sequential(*[ResBlock(c4) for _ in range(n_blocks)])
        self.up3 = up(c4, c3)
        self.dec3 = nn.Sequential(*[ResBlock(c3) for _ in range(n_blocks)])
        self.up2 = up(c3, c2)
        self.dec2 = nn.Sequential(*[ResBlock(c2) for _ in range(n_blocks)])
        self.up1 = up(c2, c1)
        self.dec1 = nn.Sequential(*[ResBlock(c1) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(c1, out_ch, 3, padding=1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (self.df - h % self.df) % self.df
        pad_w = (self.df - w % self.df) % self.df
        xin = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        def run(module, *inp):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                return checkpoint(module, *inp, use_reentrant=False)
            return module(*inp)

        h0 = self.head(xin)
        e1 = run(self.enc1, h0)
        e2 = run(self.enc2, self.down1(e1))
        e3 = run(self.enc3, self.down2(e2))
        b = run(self.body, self.down3(e3))
        d3 = run(self.dec3, self.up3(b) + e3)
        d2 = run(self.dec2, self.up2(d3) + e2)
        d1 = run(self.dec1, self.up1(d2) + e1)
        out = self.tail(d1 + h0)
        if self.residual:
            out = out + xin
        return out[..., :h, :w]
