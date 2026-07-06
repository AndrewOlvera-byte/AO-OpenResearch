"""Invalid-action masking: the masked head/policy must never pick a masked-out
action, and ``sample`` / ``log_prob_entropy`` must agree over the *same* mask (a
correctness requirement for the PPO importance ratio)."""

import torch

from models.masked_policy import MaskedActorPolicy
from models.shared.MaskedActionHead import MaskedMultiDiscreteActionHead

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
COMP_SPLITS = [6, 4, 4, 4, 4, 7, 49]
MASK_WIDTH = 1 + sum(COMP_SPLITS)  # 79
OBS = (27, 16, 16)
FEAT_C = 16


def make_mask(valid_sources, seed=0):
    """(N, 256, 79) binary mask. ``valid_sources[i]`` is the list of legal source
    cells for row i; each legal cell gets exactly one legal index per component."""
    g = torch.Generator().manual_seed(seed)
    n = len(valid_sources)
    mask = torch.zeros(n, 256, MASK_WIDTH)
    for i, cells in enumerate(valid_sources):
        for c in cells:
            mask[i, c, 0] = 1.0
            off = 1
            for sz in COMP_SPLITS:
                k = int(torch.randint(sz, (1,), generator=g).item())
                mask[i, c, off + k] = 1.0
                off += sz
    return mask


def assert_action_is_legal(action, mask):
    for i in range(action.shape[0]):
        src = int(action[i, 0])
        assert mask[i, src, 0] > 0.5, f"row {i}: picked illegal source {src}"
        off = 1
        for j, sz in enumerate(COMP_SPLITS):
            comp = int(action[i, 1 + j])
            assert mask[i, src, off + comp] > 0.5, f"row {i}: illegal comp {j}={comp}"
            off += sz


def test_sampled_action_respects_mask():
    torch.manual_seed(0)
    head = MaskedMultiDiscreteActionHead(FEAT_C, NVEC)
    feat = torch.randn(6, FEAT_C, 16, 16)
    mask = make_mask([[17, 34], [3], [200, 201, 202], [0], [17, 88, 130], [255]])
    action, logp = head.sample(feat, mask)
    assert action.shape == (6, 8) and logp.shape == (6,)
    assert_action_is_legal(action, mask)


def test_sample_eval_logprob_consistent():
    torch.manual_seed(1)
    head = MaskedMultiDiscreteActionHead(FEAT_C, NVEC)
    feat = torch.randn(5, FEAT_C, 16, 16)
    mask = make_mask([[17, 34, 51], [3, 4], [200], [10, 20], [88]])
    action, logp = head.sample(feat, mask)
    logp2, ent = head.log_prob_entropy(feat, action, mask)
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert ent.shape == (5,) and torch.isfinite(ent).all()


def test_invalid_source_has_zero_probability():
    head = MaskedMultiDiscreteActionHead(FEAT_C, NVEC)
    feat = torch.randn(2, FEAT_C, 16, 16)
    mask = make_mask([[17, 34], [200]])
    probs = head._source_dist(feat, mask).dist.probs
    # Row 0: only cells 17, 34 legal -> all others ~0.
    illegal = torch.ones(256, dtype=torch.bool)
    illegal[[17, 34]] = False
    assert probs[0][illegal].max() < 1e-6
    assert probs[0][[17, 34]].sum() > 1 - 1e-4


def test_masking_bites_entropy():
    # Each legal cell has exactly one legal index per component (make_mask), so every
    # component is a single-choice (entropy 0) and irrelevant components contribute 0.
    # Total entropy is then just the source entropy over the few legal cells — far
    # below the ~18.7-nat unmasked maximum. This is the whole point of masking.
    head = MaskedMultiDiscreteActionHead(FEAT_C, NVEC)
    feat = torch.randn(3, FEAT_C, 16, 16)
    mask = make_mask([[17, 34], [200], [10, 20, 30, 40]])
    action, _ = head.sample(feat, mask)
    _, ent = head.log_prob_entropy(feat, action, mask)
    # <= log(#legal sources) + small; nowhere near the unmasked max.
    assert (ent <= torch.log(torch.tensor([2.0, 1.0, 4.0])) + 1e-4).all()
    assert float(ent.max()) < 2.0


def test_no_actable_unit_does_not_nan():
    head = MaskedMultiDiscreteActionHead(FEAT_C, NVEC)
    feat = torch.randn(2, FEAT_C, 16, 16)
    mask = make_mask([[17], []])  # row 1 has no legal source at all
    action, logp = head.sample(feat, mask)
    assert torch.isfinite(logp).all()
    logp2, ent = head.log_prob_entropy(feat, action, mask)
    assert torch.isfinite(logp2).all() and torch.isfinite(ent).all()


def test_masked_policy_step_and_evaluate():
    torch.manual_seed(2)
    policy = MaskedActorPolicy(OBS, NVEC, device="cpu")
    obs = torch.randn(4, *OBS)
    mask = make_mask([[17, 34], [3], [200, 201], [17]])
    out = policy.step(obs, mask)
    assert tuple(out["action"].shape) == (4, 8) and out["value"].shape == (4,)
    assert_action_is_legal(out["action"], mask)

    logp, ent, value = policy.evaluate_actions(obs, out["action"], mask)
    assert torch.allclose(logp, out["logprob"], atol=1e-5)
    assert torch.isfinite(ent).all() and torch.isfinite(value).all()


def test_masked_policy_greedy_is_legal():
    policy = MaskedActorPolicy(OBS, NVEC, device="cpu")
    obs = torch.randn(3, *OBS)
    mask = make_mask([[17, 34], [200], [88, 130]])
    out = policy.step(obs, mask, deterministic=True)
    assert_action_is_legal(out["action"], mask)
