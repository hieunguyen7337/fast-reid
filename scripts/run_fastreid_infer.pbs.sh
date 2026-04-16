#!/bin/bash -l
#PBS -N fastreid_infer
#PBS -l walltime=02:00:00
#PBS -l mem=16gb
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

set -euo pipefail

echo '================================================'
WORKSPACE="${PBS_O_WORKDIR:-$(pwd)}"
echo "Submission directory = ${WORKSPACE}"
echo '================================================'
cd "${WORKSPACE}"

if [[ $(basename "$PWD") == "scripts" ]]; then
    cd ..
fi

echo "Working directory is now: $(pwd)"

echo '=========='
echo 'Fixed job configuration'
echo '=========='
CONDA_ENV_NAME="fastreid"
CONFIG_FILE="configs/Market1501/sbs_R101-ibn.yml"
WEIGHTS_PATH="weights/market1501/market_sbs_R101-ibn.pth"
INPUT_PATH="test_data"
OUTPUT_PATH="outputs/test_data_reid_embeddings.pt"
DEVICE="cuda"
BATCH_SIZE="32"
INPUT_MODE="tracklets"
SAVE_FRAME_EMBEDDINGS="1"
GPU_INDEX="0"

echo '=========='
echo 'Load CUDA & cuDNN modules'
echo '=========='
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

echo '=========='
echo 'Activate conda env'
echo '=========='
source ~/miniconda3/etc/profile.d/conda.sh
echo "Activating conda env: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

echo '=========='
echo 'Environment diagnostics'
echo '=========='
nvidia-smi || true
which python

python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
EOF

echo '=========='
echo 'Prepare inference arguments'
echo '=========='

if [ ! -s "tools/infer_reid_embeddings.py" ]; then
  echo "ERROR: tools/infer_reid_embeddings.py was not found in $(pwd)."
  exit 1
fi

if [ ! -s "${WEIGHTS_PATH}" ]; then
  echo "ERROR: checkpoint not found at ${WEIGHTS_PATH}"
  exit 1
fi

if [ ! -e "${INPUT_PATH}" ]; then
  echo "ERROR: INPUT_PATH does not exist: ${INPUT_PATH}"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

echo "Using GPU index: ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${WEIGHTS_PATH}"
echo "Input path: ${INPUT_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Input mode: ${INPUT_MODE}"
echo "Batch size: ${BATCH_SIZE}"

echo '========================='
echo 'Running FastReID inference'
echo '========================='
date

EXTRA_ARGS=()
if [ "${SAVE_FRAME_EMBEDDINGS}" = "1" ]; then
  EXTRA_ARGS+=(--save-frame-embeddings)
fi

python tools/infer_reid_embeddings.py \
  "${INPUT_PATH}" \
  --config-file "${CONFIG_FILE}" \
  --weights "${WEIGHTS_PATH}" \
  --output "${OUTPUT_PATH}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --input-mode "${INPUT_MODE}" \
  "${EXTRA_ARGS[@]}"

echo '========================='
echo 'Done.'
echo '========================='
date
