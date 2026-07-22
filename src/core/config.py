from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
import yaml
import re


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def interpolate_variables(cfg: dict) -> dict:
    pattern = r'\$\{([^}]+)\}'

    def resolve_value(value, cfg_root):
        if isinstance(value, str):
            def replace_var(match):
                path = match.group(1)
                keys = path.split('.')
                result = cfg_root
                try:
                    for key in keys:
                        result = result[key]
                    return str(result)
                except (KeyError, TypeError):
                    return match.group(0)

            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: resolve_value(v, cfg_root) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item, cfg_root) for item in value]
        else:
            return value

    return resolve_value(cfg, cfg)


def _parse_scalar(raw: str):
    """Parse a CLI override value. YAML handles bools/lists/null; we additionally
    coerce numeric strings YAML leaves as text (e.g. ``1e-3`` lacks a dot so
    PyYAML keeps it a string)."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


@dataclass
class Config:
    run: Dict[str, Any]
    model: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    wandb: Dict[str, Any]

    @classmethod
    def load(cls, path: str | Path, base_configs: List[str | Path] = None):
        merged = {}

        if base_configs:
            for base_path in base_configs:
                if Path(base_path).exists():
                    with open(base_path, "r") as f:
                        base = yaml.safe_load(f)
                    merged = deep_merge(merged, base)

        with open(path, "r") as f:
            main = yaml.safe_load(f)

        merged = deep_merge(merged, main)

        merged = interpolate_variables(merged)

        known_fields = {
            'run',
            'model',
            'data',
            'training',
            'wandb',
        }
        filtered = {k: v for k, v in merged.items() if k in known_fields}

        return cls(**filtered)

    def apply_overrides(self, overrides: List[str]) -> "Config":
        """Apply ``a.b.c=value`` CLI overrides in place (value parsed as YAML).

        The first path segment selects a top-level section (run/model/data/
        training/wandb); intermediate dicts are created as needed. Example::

            cfg.apply_overrides(["training.ppo.lr=1e-3", "run.seed=7"])
        """
        for item in overrides or []:
            key, sep, raw = item.partition("=")
            if not sep:
                raise ValueError(f"override must be key=value, got {item!r}")
            value = _parse_scalar(raw)  # 'true'->bool, '1e-3'->float, '[a,b]'->list
            parts = key.split(".")
            section = getattr(self, parts[0], None)
            if not isinstance(section, dict):
                raise KeyError(f"unknown config section {parts[0]!r} in override {item!r}")
            node = section
            for p in parts[1:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return self

    @classmethod
    def from_experiment(cls, exp_name: str):
        # Resolve config paths relative to the configs/ root (src/configs), not the
        # process CWD, so the entrypoint works from anywhere.
        configs_root = Path(__file__).resolve().parent.parent / "configs"
        common_path = configs_root / "base" / "common.yaml"
        exp_path = configs_root / "exp" / f"{exp_name}.yaml"
        return cls.load(exp_path, base_configs=[common_path])
