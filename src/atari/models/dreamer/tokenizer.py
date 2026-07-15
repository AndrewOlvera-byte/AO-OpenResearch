"""``AtariTokenizer`` — DreamerV4 causal tokenizer for 64x64 Atari frames.

A small convolutional autoencoder: four stride-2 conv stages take ``(C,64,64) ->
(ch,4,4)``, a 1x1 conv projects to ``d_latent`` channels, and flattening the ``4x4``
grid gives ``n_spatial = 16`` latent tokens (``tanh``-squashed like Dreamer 4). The
decoder mirrors this back to a ``(C,64,64)`` reconstruction with a ``sigmoid`` output
(frames are scaled to ``[0,1]``). No mask head (Atari has no action mask).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TokenizerConfig


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class AtariTokenizer(nn.Module):
    def __init__(self, obs_shape, cfg: TokenizerConfig) -> None:
        super().__init__()
        c, h, w = obs_shape
        assert cfg.downsample == 16, "AtariTokenizer assumes a /16 bottleneck (four stride-2)"
        assert h % 16 == 0 and w % 16 == 0, "obs H,W must be divisible by 16"
        self.cfg = cfg
        self.obs_channels = c
        self.bottleneck_hw = (h // 16, w // 16)
        self.n_spatial = (h // 16) * (w // 16)
        self.d_latent = cfg.d_latent

        b = cfg.base_channels
        chs = (b, b * 2, b * 4, b * 4)
        enc, prev = [], c
        for ch in chs:
            enc += [nn.Conv2d(prev, ch, 4, stride=2, padding=1), _group_norm(ch), nn.SiLU()]
            prev = ch
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Conv2d(prev, cfg.d_latent, 1)

        self.from_latent = nn.Conv2d(cfg.d_latent, prev, 1)
        dec = [_group_norm(prev), nn.SiLU()]
        rev = (b * 4, b * 2, b, b)
        cur = prev
        for ch in rev:
            dec += [nn.ConvTranspose2d(cur, ch, 4, stride=2, padding=1), _group_norm(ch), nn.SiLU()]
            cur = ch
        dec += [nn.Conv2d(cur, c, 1)]
        self.decoder = nn.Sequential(*dec)

    @staticmethod
    def _flatten_time(x: torch.Tensor, core_ndim: int):
        if x.dim() == core_ndim + 2:
            b, t = x.shape[:2]
            return x.reshape(b * t, *x.shape[2:]), (b, t)
        return x, None

    @staticmethod
    def _restore_time(x: torch.Tensor, bt):
        if bt is None:
            return x
        b, t = bt
        return x.reshape(b, t, *x.shape[1:])

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        x, bt = self._flatten_time(obs, core_ndim=3)
        z = self.to_latent(self.encoder(x))                  # (N, d_latent, h/16, w/16)
        n = z.shape[0]
        z = z.permute(0, 2, 3, 1).reshape(n, self.n_spatial, self.d_latent)
        if self.cfg.tanh_bottleneck:
            z = torch.tanh(z)
        return self._restore_time(z, bt)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x, bt = self._flatten_time(z, core_ndim=2)
        n = x.shape[0]
        hb, wb = self.bottleneck_hw
        m = x.reshape(n, hb, wb, self.d_latent).permute(0, 3, 1, 2)
        recon = torch.sigmoid(self.decoder(self.from_latent(m)))
        return self._restore_time(recon, bt)

    def forward(self, obs: torch.Tensor):
        z = self.encode(obs)
        return self.decode(z), z
