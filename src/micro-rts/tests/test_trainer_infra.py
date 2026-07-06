"""Checkpointing, best tracking, eval-history JSON, and collapse guards."""

import json

import torch

from core.config import Config
from models.cnn_mlp_policy import CNNMLPPolicy
from trainers.BaseTrainer import BaseTrainer, CollapseError
from trainers.guards import entropy_fraction, explained_variance, is_finite, max_entropy

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
OBS = (27, 16, 16)


def _trainer(tmp_path):
    cfg = Config.from_experiment("micro-rts/base_rlFS_expert")
    cfg.run["ckpt_dir"] = str(tmp_path)
    t = BaseTrainer(cfg, device="cpu")
    return t


# --- guards (pure) -------------------------------------------------------
def test_is_finite_tree():
    assert is_finite({"a": 1.0, "b": torch.tensor([1.0, 2.0])})
    assert not is_finite({"a": float("nan")})
    assert not is_finite(torch.tensor([1.0, float("inf")]))


def test_explained_variance_perfect_and_useless():
    r = torch.randn(100)
    assert explained_variance(r, r) > 0.99               # perfect prediction
    assert explained_variance(torch.zeros(100), r) < 0.01  # predicting the mean


def test_entropy_fraction():
    m = max_entropy(NVEC)
    assert abs(entropy_fraction(m, NVEC) - 1.0) < 1e-6
    assert abs(entropy_fraction(0.0, NVEC)) < 1e-6


# --- checkpoint + best + eval history ------------------------------------
def test_checkpoint_roundtrip_includes_optimizer_and_actor_critic(tmp_path):
    t = _trainer(tmp_path)
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    opt = torch.optim.Adam(policy.parameters())
    path = t.save_checkpoint(policy, opt, step=10, metrics={"loss": 0.5})
    payload = torch.load(path, weights_only=False)
    assert payload["step"] == 10 and payload["optimizer"] is not None
    keys = payload["model"].keys()
    assert any(k.startswith("actor.") for k in keys)
    assert any(k.startswith("critic.") for k in keys)
    # round-trips back into a fresh policy + optimizer.
    fresh = CNNMLPPolicy(OBS, NVEC, device="cpu")
    fresh_opt = torch.optim.Adam(fresh.parameters())
    t.load_checkpoint(fresh, fresh_opt)
    for a, b in zip(policy.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a, b)


def test_checkpoint_rotation(tmp_path):
    t = _trainer(tmp_path)
    t.keep_last = 2
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    opt = torch.optim.Adam(policy.parameters())
    for step in (1, 2, 3, 4):
        t.save_checkpoint(policy, opt, step=step, tag=f"step_{step}")
    remaining = sorted(p.name for p in t.run_dir.glob("step_*.pt"))
    assert remaining == ["step_3.pt", "step_4.pt"]


def test_record_eval_best_tracking_and_history(tmp_path):
    t = _trainer(tmp_path)
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    opt = torch.optim.Adam(policy.parameters())
    assert t.record_eval(10, {"eval/win_rate": 0.1}, policy, opt) is True
    assert t.record_eval(20, {"eval/win_rate": 0.05}, policy, opt) is False  # worse
    assert t.record_eval(30, {"eval/win_rate": 0.3}, policy, opt) is True   # better
    history = json.loads((t.run_dir / "eval_history.json").read_text())
    assert [h["step"] for h in history] == [10, 20, 30]
    assert [h["is_best"] for h in history] == [True, False, True]
    assert (t.run_dir / "best.pt").exists()
    best = json.loads((t.run_dir / "best.json").read_text())
    assert best["step"] == 30


def test_guard_finite_raises(tmp_path):
    t = _trainer(tmp_path)
    t.guard_finite({"loss": 1.0}, step=0)  # ok
    try:
        t.guard_finite({"loss": float("nan")}, step=1)
        assert False, "expected CollapseError"
    except CollapseError:
        pass
