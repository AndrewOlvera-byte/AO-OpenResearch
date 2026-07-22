from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EgoTokenizerConfig


OBS_GROUPS = (5, 5, 3, 8, 6)


class EgoObservationTokenizer(nn.Module):
    """Fog-aware local encoder; hidden-world inference intentionally lives later."""

    def __init__(self, grid_hw=(16, 16), cfg: EgoTokenizerConfig | None = None):
        super().__init__()
        self.cfg = cfg or EgoTokenizerConfig()
        self.h, self.w = map(int, grid_hw)
        ds, d = int(self.cfg.downsample), int(self.cfg.d_model)
        if self.h % ds or self.w % ds:
            raise ValueError("grid must be divisible by ego tokenizer downsample")
        self.n_spatial = (self.h // ds) * (self.w // ds)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.cfg.obs_channels + 1, d // 2, 3, padding=1),
            nn.GroupNorm(8 if (d // 2) % 8 == 0 else 1, d // 2),
            nn.SiLU(),
            nn.Conv2d(d // 2, d, 3, stride=ds, padding=1),
            nn.GroupNorm(8 if d % 8 == 0 else 1, d),
            nn.SiLU(),
        )
        self.visibility_embed = nn.Linear(1, d)
        self.spatial_position = nn.Parameter(torch.zeros(self.n_spatial, d))
        self.registers = nn.Parameter(torch.zeros(self.cfg.n_registers, d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.normal_(self.registers, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d,
            self.cfg.n_heads,
            int(d * self.cfg.mlp_ratio),
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, self.cfg.depth, nn.LayerNorm(d))
        self.to_latent = nn.Linear(d, self.cfg.d_latent)
        self.from_latent = nn.Conv2d(self.cfg.d_latent, d, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d, d // 2, 4, stride=ds, padding=1),
            nn.GroupNorm(8 if (d // 2) % 8 == 0 else 1, d // 2),
            nn.SiLU(),
        )
        self.obs_head = nn.Conv2d(d // 2, sum(OBS_GROUPS), 1)
        self.visibility_head = nn.Conv2d(d // 2, 1, 1)

    def encode(self, obs, visibility, patch_mask: torch.Tensor | None = None):
        lead = obs.shape[:-3]
        flat_obs = obs.reshape(-1, *obs.shape[-3:]).float()
        flat_vis = visibility.reshape(-1, *visibility.shape[-3:]).float()
        feat = self.encoder(torch.cat((flat_obs, flat_vis), dim=1))
        spatial = feat.flatten(2).transpose(1, 2)
        vis_block = (
            F.avg_pool2d(flat_vis, self.cfg.downsample, self.cfg.downsample)
            .flatten(2)
            .transpose(1, 2)
        )
        spatial = spatial + self.visibility_embed(vis_block) + self.spatial_position
        if patch_mask is not None:
            mask = patch_mask.reshape(-1, self.n_spatial).bool()
            spatial = torch.where(
                mask[..., None], self.mask_token.view(1, 1, -1), spatial
            )
        registers = self.registers[None].expand(spatial.shape[0], -1, -1)
        encoded = self.transformer(torch.cat((registers, spatial), dim=1))
        registers = self.to_latent(encoded[:, : self.cfg.n_registers])
        spatial = self.to_latent(encoded[:, self.cfg.n_registers :])
        return (
            spatial.reshape(*lead, self.n_spatial, self.cfg.d_latent),
            registers.reshape(*lead, self.cfg.n_registers, self.cfg.d_latent),
            vis_block.reshape(*lead, self.n_spatial, 1),
        )

    def decode(self, spatial):
        lead = spatial.shape[:-2]
        flat = spatial.reshape(-1, self.n_spatial, self.cfg.d_latent)
        grid = flat.transpose(1, 2).reshape(
            flat.shape[0],
            self.cfg.d_latent,
            self.h // self.cfg.downsample,
            self.w // self.cfg.downsample,
        )
        feat = self.decoder(self.from_latent(grid))
        obs_logits = self.obs_head(feat).reshape(*lead, sum(OBS_GROUPS), self.h, self.w)
        vis_logits = self.visibility_head(feat).reshape(*lead, 1, self.h, self.w)
        return {"obs_logits": obs_logits, "visibility_logits": vis_logits}

    def forward(self, obs, visibility, patch_mask=None):
        spatial, registers, vis_block = self.encode(obs, visibility, patch_mask)
        return self.decode(spatial), spatial, registers, vis_block


class EgoTokenizerPretrainer(nn.Module):
    def __init__(self, grid_hw=(16, 16), cfg: EgoTokenizerConfig | None = None):
        super().__init__()
        self.tokenizer = EgoObservationTokenizer(grid_hw, cfg)
        self.target = copy.deepcopy(self.tokenizer).requires_grad_(False)
        self.target.eval()
        d = self.tokenizer.cfg.d_latent
        self.jepa_predictor = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        decay = float(self.tokenizer.cfg.ema_decay)
        for target, source in zip(
            self.target.parameters(), self.tokenizer.parameters()
        ):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
