#!/usr/bin/env bash
# Collect physically separate, complete-episode train/validation/evaluation files,
# add factual + counterfactual fog views, and run exhaustive collection audits.
set -euo pipefail

CONTAINER="${CONTAINER:-ao-research}"
ROOT="${ROOT:-/data/micro-rts/world_action_v2}"
NUM_ENVS="${NUM_ENVS:-24}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CF_FRAC="${CF_FRAC:-0.20}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-2000}"
STRONG="${STRONG:-checkpoints/base_rlFS_expert_masked_league/best.pt}"
TRAIN_SEED="${TRAIN_SEED:-11001}"
VALIDATION_SEED="${VALIDATION_SEED:-22001}"
EVALUATION_SEED="${EVALUATION_SEED:-33001}"
SMOKE="${SMOKE:-0}"

MAPS=(
  maps/16x16/basesWorkers16x16.xml
  maps/16x16/basesWorkers16x16A.xml
  maps/16x16/TwoBasesBarracks16x16.xml
  maps/16x16/melee16x16Mixed8.xml
)
if [[ "${SMOKE}" == "1" ]]; then
  MAPS=(maps/16x16/basesWorkers16x16.xml)
fi

run_split() {
  local split="$1"
  local seed="$2"
  local scale="$3"
  shift 3
  local raw="${ROOT}/raw/${split}.h5"
  local final="${ROOT}/${split}.h5"
  local -a bots=("$@")
  local -a plans
  if [[ "${SMOKE}" == "1" ]]; then
    plans=(
      "mode=selfplay,policy=masked_random,steps=40"
      "mode=bot,policy=masked_random,steps=40,seats=mix"
    )
  elif [[ "${scale}" == "train" ]]; then
    plans=(
      "mode=selfplay,policy=${STRONG},eps=0.05,steps=4000"
      "mode=selfplay,policy=${STRONG},eps=0.15,steps=7000"
      "mode=selfplay,policy=${STRONG},eps=0.30,steps=6000"
      "mode=selfplay,policy=${STRONG},eps=0.50,steps=3000"
      "mode=selfplay,policy=masked_random,steps=4000"
      "mode=bot,policy=${STRONG},steps=5000,seats=mix"
      "mode=bot,policy=${STRONG},eps=0.15,steps=4000,seats=mix"
      "mode=bot,policy=${STRONG},eps=0.30,steps=3000,seats=mix"
      "mode=bot,policy=masked_random,steps=4000,seats=mix"
    )
  else
    # 5k steps/lane/map: one eighth of train, preserving its 60/40 mix.
    plans=(
      "mode=selfplay,policy=${STRONG},eps=0.05,steps=500"
      "mode=selfplay,policy=${STRONG},eps=0.15,steps=875"
      "mode=selfplay,policy=${STRONG},eps=0.30,steps=750"
      "mode=selfplay,policy=${STRONG},eps=0.50,steps=375"
      "mode=selfplay,policy=masked_random,steps=500"
      "mode=bot,policy=${STRONG},steps=625,seats=mix"
      "mode=bot,policy=${STRONG},eps=0.15,steps=500,seats=mix"
      "mode=bot,policy=${STRONG},eps=0.30,steps=375,seats=mix"
      "mode=bot,policy=masked_random,steps=500,seats=mix"
    )
  fi

  local -a cmd=(
    python src/micro-rts/collectors/offline_data/collect_mrts_data.py
    --name "world_action_v2_${split}"
    --output "${raw}"
    --split "${split}"
    --episode-aware
    --seed "${seed}"
    --num-envs "${NUM_ENVS}"
    --policy-device "${POLICY_DEVICE}"
    --max-episode-steps "${MAX_EPISODE_STEPS}"
    --counterfactual-frac "${CF_FRAC}"
    --gzip 1
    --chunk-rows 512
    --maps "${MAPS[@]}"
    --bots "${bots[@]}"
  )
  local plan
  for plan in "${plans[@]}"; do
    cmd+=(--plan "${plan}")
  done
  "${cmd[@]}"
  python src/micro-rts/entrypoints/util/augment_fog_observations.py \
    "${raw}" "${final}" --gzip 1 --block-rows 2048
}

run_inside() {
  cd /workspace
  if [[ "${SMOKE}" != "1" ]]; then
    test -f "${STRONG}"
  fi
  bash infra/microrts-jar-patch/apply_patch.sh
  mkdir -p "${ROOT}/raw" "${ROOT}/manifests"
  run_split train "${TRAIN_SEED}" train \
    randomBiasedAI workerRushAI lightRushAI
  run_split validation "${VALIDATION_SEED}" heldout \
    randomBiasedAI workerRushAI lightRushAI
  # Matched opponents plus coacAI as an evaluation-only generalization stratum.
  run_split evaluation "${EVALUATION_SEED}" heldout \
    randomBiasedAI workerRushAI lightRushAI coacAI
  python src/micro-rts/entrypoints/util/audit_world_action_collection.py \
    "${ROOT}/train.h5" "${ROOT}/validation.h5" "${ROOT}/evaluation.h5" \
    --json "${ROOT}/manifests/audit.json"
}

if [[ "${INSIDE:-0}" == "1" ]]; then
  run_inside
else
  exec docker exec -i \
    -e INSIDE=1 -e ROOT="${ROOT}" -e NUM_ENVS="${NUM_ENVS}" \
    -e POLICY_DEVICE="${POLICY_DEVICE}" -e CF_FRAC="${CF_FRAC}" \
    -e MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" -e STRONG="${STRONG}" \
    -e TRAIN_SEED="${TRAIN_SEED}" -e VALIDATION_SEED="${VALIDATION_SEED}" \
    -e EVALUATION_SEED="${EVALUATION_SEED}" \
    -e SMOKE="${SMOKE}" \
    "${CONTAINER}" bash -lc "cd /workspace && ./scripts/collect_world_action_v2.sh"
fi
