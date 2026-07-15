"""``GridTokenizer`` — the Dreamer 4 causal tokenizer, realized as the GridNet CNN.

Dreamer 4's tokenizer is an autoencoder that squeezes each frame through a small
continuous latent and reconstructs it. Rather than the patch-transformer tokenizer
of the vendored reference (tuned for 128x128 RGB video), we reuse the
**encoder/decoder CNN of the canonical Gym-uRTS GridNet policy** — the exact trunk
that already learns well on the 16x16 MicroRTS obs grid:

- **encoder**: two stride-2 convs take ``(C, H, W) -> (enc_channels, H/4, W/4)``;
  a 1x1 conv projects to ``d_latent`` channels; flattening the ``H/4 x W/4`` grid
  yields ``n_spatial`` latent tokens of width ``d_latent`` (``tanh``-squashed, as in
  Dreamer 4, to keep the world model's target latents bounded).
- **decoder**: reshapes the tokens back to a ``d_latent x H/4 x W/4`` map and
  conv-transposes back to ``(C, H, W)`` for reconstruction.

The tokens are laid out row-major so token ``i`` corresponds to bottleneck cell
``i`` — the same convention the GridNet action head uses over the full grid. Both
``encode``/``decode`` accept a leading time axis ``(B, T, ...)`` (folded internally)
or a plain batch ``(B, ...)``.
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


class GridTokenizer(nn.Module):
    def __init__(self, obs_shape, cfg: TokenizerConfig, mask_width: int | None = None) -> None:
        super().__init__()
        c, h, w = obs_shape
        assert cfg.downsample == 4, "GridTokenizer currently assumes a /4 bottleneck"
        assert h % 4 == 0 and w % 4 == 0, "obs H,W must be divisible by 4"
        self.cfg = cfg
        self.obs_channels = c
        self.grid_hw = (h, w)
        self.grid_cells = h * w
        self.mask_width = mask_width
        self.bottleneck_hw = (h // 4, w // 4)
        self.n_spatial = (h // 4) * (w // 4)
        self.d_latent = cfg.d_latent

        ch = cfg.enc_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), _group_norm(32), nn.ReLU(),
            nn.Conv2d(32, ch, 3, padding=1, stride=2), _group_norm(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, stride=2), _group_norm(ch), nn.ReLU(),
        )
        self.to_latent = nn.Conv2d(ch, cfg.d_latent, 1)

        self.from_latent = nn.Conv2d(cfg.d_latent, ch, 1)
        self.decoder = nn.Sequential(
            _group_norm(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), _group_norm(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, 32, 4, stride=2, padding=1), _group_norm(32), nn.ReLU(),
            nn.Conv2d(32, c, 1),
        )

        # Predicted action-mask head: decodes the latent to per-cell mask logits so
        # imagination (no engine) can still mask its GridNet actions. Trained with
        # BCE against the real engine mask in the world-model loss. Lives on the
        # tokenizer so it optimizes with the world model, not the actor.
        if mask_width is not None:
            self.mask_from_latent = nn.Conv2d(cfg.d_latent, ch, 1)
            self.mask_decoder = nn.Sequential(
                _group_norm(ch), nn.ReLU(),
                nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), _group_norm(ch), nn.ReLU(),
                nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), _group_norm(ch), nn.ReLU(),
                nn.Conv2d(ch, mask_width, 1),
            )

    # --- shape helpers ---------------------------------------------------
    @staticmethod
    def _flatten_time(x: torch.Tensor, core_ndim: int):
        """Fold a leading time axis when present.

        ``core_ndim`` is the per-item rank (obs: 3 for C,H,W; latent tokens: 2 for
        n_spatial,d_latent). ``(B,T,*core) -> ((B*T,*core), (B,T))``; a plain batch
        ``(B,*core)`` passes through with ``None``.
        """
        if x.dim() == core_ndim + 2:
            b, t = x.shape[:2]
            return x.reshape(b * t, *x.shape[2:]), (b, t)
        return x, None

    @staticmethod
    def _restore_time(x: torch.Tensor, bt) -> torch.Tensor:
        if bt is None:
            return x
        b, t = bt
        return x.reshape(b, t, *x.shape[1:])

    # --- encode / decode -------------------------------------------------
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B,[T],C,H,W) -> latent tokens (B,[T],n_spatial,d_latent)."""
        x, bt = self._flatten_time(obs, core_ndim=3)
        z = self.to_latent(self.encoder(x))                 # (N, d_latent, h/4, w/4)
        n = z.shape[0]
        z = z.permute(0, 2, 3, 1).reshape(n, self.n_spatial, self.d_latent)
        if self.cfg.tanh_bottleneck:
            z = torch.tanh(z)
        return self._restore_time(z, bt)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """latent tokens (B,[T],n_spatial,d_latent) -> obs recon (B,[T],C,H,W)."""
        x, bt = self._flatten_time(z, core_ndim=2)
        n = x.shape[0]
        hb, wb = self.bottleneck_hw
        m = x.reshape(n, hb, wb, self.d_latent).permute(0, 3, 1, 2)  # (N,d_latent,h/4,w/4)
        recon = self.decoder(self.from_latent(m))                    # (N, C, H, W)
        return self._restore_time(recon, bt)

    def decode_mask(self, z: torch.Tensor) -> torch.Tensor:
        """latent tokens (B,[T],n_spatial,d_latent) -> mask logits (B,[T],H*W,mask_width)."""
        assert self.mask_width is not None, "tokenizer built without a mask head"
        x, bt = self._flatten_time(z, core_ndim=2)
        n = x.shape[0]
        hb, wb = self.bottleneck_hw
        m = x.reshape(n, hb, wb, self.d_latent).permute(0, 3, 1, 2)
        logits = self.mask_decoder(self.mask_from_latent(m))          # (N, mask_width, H, W)
        logits = logits.permute(0, 2, 3, 1).reshape(n, self.grid_cells, self.mask_width)
        return self._restore_time(logits, bt)

    def forward(self, obs: torch.Tensor):
        z = self.encode(obs)
        return self.decode(z), z
