#!/usr/bin/env bash
# Collect the Markov-complete MicroRTS world-model v2 corpus.
#
# Default budget:
#   40k steps/lane/map x 24 lanes x 4 maps = 3.84M real transitions
#   20% paired cloned-engine branches       = ~768k counterfactual transitions
#
# The mix is 60% self-play, where both action streams receive independent
# policy/noise variation, and 40% scripted-bot anchoring. Four 16x16 maps cover
# economy, production, barracks, and dense combat while preserving one model
# shape. coacAI remains held out for evaluation.
#
# Host usage:
#   ./scripts/collect_wm_v2.sh
#
# Useful overrides:
#   NUM_ENVS=32 CF_FRAC=0.25 RUN_NAME=wm_v2_large ./scripts/collect_wm_v2.sh
#   STRONG=checkpoints/other/best.pt ./scripts/collect_wm_v2.sh
#
# Inside the container:
#   INSIDE=1 ./scripts/collect_wm_v2.sh

set -euo pipefail

CONTAINER="${CONTAINER:-ao-research}"
RUN_NAME="${RUN_NAME:-wm_v2_pretrain}"
OUT_DIR="${OUT_DIR:-/data/micro-rts}"
NUM_ENVS="${NUM_ENVS:-24}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CF_FRAC="${CF_FRAC:-0.20}"
STRONG="${STRONG:-checkpoints/base_rlFS_expert_masked_league/best.pt}"

MAPS=(
  maps/16x16/basesWorkers16x16.xml
  maps/16x16/basesWorkers16x16A.xml
  maps/16x16/TwoBasesBarracks16x16.xml
  maps/16x16/melee16x16Mixed8.xml
)

CMD=(python src/micro-rts/collectors/offline_data/collect_mrts_data.py
  --name "${RUN_NAME}"
  --out-dir "${OUT_DIR}"
  --num-envs "${NUM_ENVS}"
  --policy-device "${POLICY_DEVICE}"
  --counterfactual-frac "${CF_FRAC}"
  --gzip 1
  --chunk-rows 512
  --maps "${MAPS[@]}"
  # --- self-play 60%: strongest joint-action identifiability -------------
  --plan "mode=selfplay,policy=${STRONG},eps=0.05,steps=4000"
  --plan "mode=selfplay,policy=${STRONG},eps=0.15,steps=7000"
  --plan "mode=selfplay,policy=${STRONG},eps=0.30,steps=6000"
  --plan "mode=selfplay,policy=${STRONG},eps=0.50,steps=3000"
  --plan "mode=selfplay,policy=masked_random,steps=4000"
  # --- bot 40%: realistic strategy and seat-role anchors -----------------
  --plan "mode=bot,policy=${STRONG},steps=5000,seats=mix"
  --plan "mode=bot,policy=${STRONG},eps=0.15,steps=4000,seats=mix"
  --plan "mode=bot,policy=${STRONG},eps=0.30,steps=3000,seats=mix"
  --plan "mode=bot,policy=masked_random,steps=4000,seats=mix"
)

REAL_TRANSITIONS=$((40000 * NUM_ENVS * ${#MAPS[@]}))
PAIRED_BRANCHES=$(awk -v n="${REAL_TRANSITIONS}" -v f="${CF_FRAC}" \
  'BEGIN { printf "%.0f", n * f }')

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[collect-v2] real=${REAL_TRANSITIONS} paired≈${PAIRED_BRANCHES}"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

run_inside() {
  cd /workspace
  if [[ ! -f "${STRONG}" ]]; then
    echo "[collect-v2] missing policy checkpoint: ${STRONG}" >&2
    exit 1
  fi
  bash infra/microrts-jar-patch/apply_patch.sh
  echo "[collect-v2] real=${REAL_TRANSITIONS} paired≈${PAIRED_BRANCHES}"
  exec "${CMD[@]}"
}

if [[ "${INSIDE:-0}" == "1" ]]; then
  run_inside
else
  printf -v QUOTED_CMD '%q ' "${CMD[@]}"
  exec docker exec -i \
    -e STRONG="${STRONG}" \
    "${CONTAINER}" bash -lc \
    "cd /workspace && \
     test -f $(printf '%q' "${STRONG}") && \
     bash infra/microrts-jar-patch/apply_patch.sh && \
     echo '[collect-v2] real=${REAL_TRANSITIONS} paired≈${PAIRED_BRANCHES}' && \
     ${QUOTED_CMD}"
fi
