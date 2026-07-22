#!/usr/bin/env bash
# Collect the tokdyn_pretrain_v3_1 corpus — the opponent-identifiability fix
# (NOTEBOOK.md "v4.4 findings -> Next steps").
#
# v3's flaw: the opponent was a deterministic scripted bot in ~85% of the data,
# so the opponent-action channel carried ~zero information beyond the state and
# the world model (correctly) learned to ignore it. v3.1 raises selfplay to 50%
# with an eps ladder — in selfplay the opponent seat is the Python policy, the
# only place opponent-side exogenous noise is injectable (scripted bots act
# inside the JVM) — plus a fully-random selfplay block, with bot blocks kept as
# distribution anchors. coacAI stays held out (eval-only), as always.
#
# 60k steps/lane x 24 lanes = 1.44M transitions (same budget as v3).
# Output: /data/micro-rts/tokdyn_pretrain_v3_1__<UTC>__<git8>.h5
#
# Run from the host (repo root or anywhere):
#   ./scripts/collect_v3_1.sh
# Already inside the container? INSIDE=1 ./scripts/collect_v3_1.sh

set -euo pipefail

STRONG=checkpoints/base_rlFS_expert_masked_league/best.pt  # same as the v3 store

CMD=(python src/micro-rts/collectors/offline_data/collect_mrts_data.py
  --name tokdyn_pretrain_v3_1
  --num-envs 24
  --policy-device cuda
  # --- selfplay 50%: the opponent channel gets the eps ladder -------------
  --plan "mode=selfplay,policy=${STRONG},eps=0.05,steps=9000"
  --plan "mode=selfplay,policy=${STRONG},eps=0.15,steps=12000"
  --plan "mode=selfplay,policy=${STRONG},eps=0.30,steps=6000"
  --plan "mode=selfplay,policy=masked_random,steps=3000"
  # --- bot 50%: realism anchors (self channel keeps its noise ladder) -----
  --plan "mode=bot,policy=${STRONG},steps=12000,seats=mix"
  --plan "mode=bot,policy=${STRONG},eps=0.15,steps=9000,seats=mix"
  --plan "mode=bot,policy=${STRONG},eps=0.30,steps=3000"
  --plan "mode=bot,policy=masked_random,steps=6000"
)

if [[ "${INSIDE:-0}" == "1" ]]; then
  cd /workspace
  exec "${CMD[@]}"
else
  exec docker exec -i ao-research bash -c "cd /workspace && $(printf '%q ' "${CMD[@]}")"
fi
