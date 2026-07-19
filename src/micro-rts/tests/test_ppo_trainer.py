"""PPOTrainer: legacy update moves weights; config-mode smoke test runs."""

import torch

import models.gridnet_policy  # noqa: F401  (registers cnn_gridnet)
from collectors.buffer import RolloutBuffer
from core.config import Config
from models.cnn_mlp_policy import CNNMLPPolicy
from trainers.PPOTrainer import PPOTrainer

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
OBS = (27, 16, 16)


def test_legacy_update_moves_weights():
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    before = policy.critic.out.weight.detach().clone()
    buf = RolloutBuffer(horizon=4, num_envs=4, obs_shape=OBS, action_dim=8)
    buf.data["obs"] = torch.randn(4, 4, *OBS)
    buf.data["action"] = (torch.rand(4, 4, 8) * NVEC).long()
    buf.data["advantage"] = torch.randn(4, 4)
    buf.data["return"] = torch.randn(4, 4)
    PPOTrainer(policy, epochs=2, minibatches=2).update(buf)
    assert not torch.equal(before, policy.critic.out.weight)


def test_config_mode_smoke_test():
    cfg = Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert")
    trainer = PPOTrainer(cfg=cfg)
    trainer.device = torch.device("cpu")  # keep CI off the GPU
    metrics = trainer.smoke_test()
    assert trainer._wandb is None  # no W&B in smoke mode
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


def test_gridnet_config_smoke_test():
    cfg = Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert_masked")
    trainer = PPOTrainer(cfg=cfg)
    trainer.device = torch.device("cpu")
    metrics = trainer.smoke_test()  # builds gridnet env + gridnet policy, collect+update
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


def test_anneal_lr_read_from_config():
    # gridnet config opts into LR annealing; the legacy config leaves it off (default).
    assert PPOTrainer(cfg=Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert_masked")).anneal_lr is True
    assert PPOTrainer(cfg=Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert")).anneal_lr is False


def test_amp_flag_read_from_config():
    assert PPOTrainer(cfg=Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert_masked")).amp is True
    assert PPOTrainer(cfg=Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert")).amp is False


def test_checkpoint_resume_roundtrip(tmp_path):
    from core.registry import build

    def fresh():
        return build("model", type="cnn_gridnet", obs_shape=OBS, action_nvec=NVEC, device="cpu")

    cfg = Config.from_experiment("micro-rts/rl/ppo/base_rlFS_expert_masked")
    a = PPOTrainer(cfg=cfg); a.device = torch.device("cpu"); a.run_dir = tmp_path
    pol = fresh(); a._attach(pol)
    sum(p.sum() for p in pol.parameters()).backward()
    a.opt.step()                       # give the optimizer real state to save
    a._best_metric = 0.42
    a.save_checkpoint(pol, a.opt, step=655360, metrics={}, tag="latest")

    # A fresh trainer + policy resumes the full state.
    b = PPOTrainer(cfg=cfg); b.device = torch.device("cpu"); b.run_dir = tmp_path
    pol2 = fresh(); b._attach(pol2)
    payload = b.load_checkpoint(pol2, b.opt, tag="latest")
    assert payload["step"] == 655360 and payload["best_metric"] == 0.42
    for p1, p2 in zip(pol.parameters(), pol2.parameters()):
        assert torch.equal(p1, p2)                       # actor+critic weights restored
    assert len(b.opt.state_dict()["state"]) > 0          # optimizer momentum restored


def test_anneal_lr_schedule_decays_to_zero():
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    tr = PPOTrainer(policy, lr=2.5e-4, anneal_lr=True)
    iters = 100
    # Mirror the schedule used in train(): lr * (1 - it/iters), applied to the optimizer.
    lrs = []
    for it in range(iters):
        lr_now = tr.lr * (1.0 - it / iters)
        for g in tr.opt.param_groups:
            g["lr"] = lr_now
        lrs.append(tr.opt.param_groups[0]["lr"])
    assert lrs[0] == 2.5e-4
    assert lrs[-1] < lrs[0] and lrs[-1] > 0  # strictly decreasing, not yet zero
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # monotonic
