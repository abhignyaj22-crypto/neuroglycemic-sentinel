#!/usr/bin/env bash
# Train a K-seed deep ensemble of the neural glucose forecaster.
#
# Usage:
#   bash scripts/train_ensemble.sh <aligned.csv.gz> <config.json> <workspace> <run-name> [members] [pretrain_epochs]
#
# Members share the data, patient split, feature schema and target
# standardization (enforced by src/neuroglycemic/ensemble.py); only the
# training seed differs between members.
set -euo pipefail

DATA=${1:?aligned dataset path required}
CONFIG=${2:?training config JSON required}
WORKSPACE=${3:?external workspace path required}
RUN_NAME=${4:?run name required}
MEMBERS=${5:-3}
PRETRAIN_EPOCHS=${6:-0}

for SEED in $(seq 42 $((41 + MEMBERS))); do
  MEMBER_CONFIG="${WORKSPACE}/runs/${RUN_NAME}/config-seed-${SEED}.json"
  mkdir -p "${WORKSPACE}/runs/${RUN_NAME}"
  python3 - "${CONFIG}" "${MEMBER_CONFIG}" "${SEED}" <<'PY'
import json, sys
source, destination, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
values = json.loads(open(source).read())
# The patient split stays keyed to the shared config seed; only model
# initialization and batch shuffling are reseeded per ensemble member.
values["optimization_seed"] = seed
open(destination, "w").write(json.dumps(values, indent=2) + "\n")
PY
  echo "=== Training ensemble member seed=${SEED} ==="
  python main.py train-neural \
    --data "${DATA}" \
    --config "${MEMBER_CONFIG}" \
    --workspace "${WORKSPACE}" \
    --run-name "${RUN_NAME}-seed-${SEED}" \
    --checkpoint "${WORKSPACE}/models/${RUN_NAME}-seed-${SEED}.pt" \
    --pretrain-epochs "${PRETRAIN_EPOCHS}" \
    --pretrain-checkpoint "${WORKSPACE}/models/${RUN_NAME}-seed-${SEED}-pretrain.pt"
done

echo "Ensemble members saved under ${WORKSPACE}/models/${RUN_NAME}-seed-*.pt"
echo "Evaluate with: python scripts/evaluate_ensemble.py --data ${DATA} --config ${CONFIG} --workspace ${WORKSPACE} --run-name ${RUN_NAME} --members ${MEMBERS}"