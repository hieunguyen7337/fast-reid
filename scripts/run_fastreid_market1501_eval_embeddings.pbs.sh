#!/bin/bash -l
#PBS -N fastreid_market1501_eval
#PBS -l walltime=04:00:00
#PBS -l mem=32gb
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

set -euo pipefail

echo "================================================"
echo "FastReID Market1501 eval embedding extraction"
echo "Submitted from: ${PBS_O_WORKDIR:-$(pwd)}"
echo "Started at: $(date)"
echo "================================================"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/hpccs01/home/n12194778/video_ai_identification}"
PLATFORM_ROOT="${PLATFORM_ROOT:-${WORKSPACE_ROOT}/AI_Agent_Local_Workflow_PLatform}"
REID_ROOT="${REID_ROOT:-${WORKSPACE_ROOT}/fast-reid}"
PARTITION_ROOT="${PARTITION_ROOT:-${PLATFORM_ROOT}/evals/person_reid_market1501/partition_1000q_5000g}"
OUTPUT_DIR="${OUTPUT_DIR:-${REID_ROOT}/outputs/person_reid_market1501_eval}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fastreid}"
CONFIG_FILE="${CONFIG_FILE:-${REID_ROOT}/configs/Market1501/sbs_R101-ibn.yml}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${REID_ROOT}/weights/market1501/market_sbs_R101-ibn.pth}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU_INDEX="${GPU_INDEX:-0}"

QUERY_INPUT="${QUERY_INPUT:-${PARTITION_ROOT}/query}"
GALLERY_INPUT="${GALLERY_INPUT:-${PARTITION_ROOT}/bounding_box_test}"
QUERY_OUTPUT="${QUERY_OUTPUT:-${OUTPUT_DIR}/fastreid_query_embeddings.pt}"
GALLERY_OUTPUT="${GALLERY_OUTPUT:-${OUTPUT_DIR}/fastreid_gallery_embeddings.pt}"

echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Platform root: ${PLATFORM_ROOT}"
echo "FastReID root: ${REID_ROOT}"
echo "Partition root: ${PARTITION_ROOT}"
echo "Query input: ${QUERY_INPUT}"
echo "Gallery input: ${GALLERY_INPUT}"
echo "Query output: ${QUERY_OUTPUT}"
echo "Gallery output: ${GALLERY_OUTPUT}"
echo "Config: ${CONFIG_FILE}"
echo "Weights: ${WEIGHTS_PATH}"
echo "Batch size: ${BATCH_SIZE}"
echo "GPU index: ${GPU_INDEX}"

if [[ ! -d "${REID_ROOT}" ]]; then
  echo "ERROR: REID_ROOT does not exist: ${REID_ROOT}" >&2
  exit 1
fi
if [[ ! -s "${REID_ROOT}/tools/infer_reid_embeddings.py" ]]; then
  echo "ERROR: inference script not found: ${REID_ROOT}/tools/infer_reid_embeddings.py" >&2
  exit 1
fi
if [[ ! -s "${CONFIG_FILE}" ]]; then
  echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -d "${QUERY_INPUT}" ]]; then
  echo "ERROR: query input directory not found: ${QUERY_INPUT}" >&2
  exit 1
fi
if [[ ! -d "${GALLERY_INPUT}" ]]; then
  echo "ERROR: gallery input directory not found: ${GALLERY_INPUT}" >&2
  exit 1
fi
if [[ ! -s "${WEIGHTS_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${WEIGHTS_PATH}" >&2
  exit 1
fi

echo "========== Load CUDA modules =========="
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

echo "========== Activate conda environment =========="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

echo "========== Environment diagnostics =========="
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
nvidia-smi || true
which python
python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
EOF

mkdir -p "${OUTPUT_DIR}"
cd "${REID_ROOT}"

echo "========== Extract query embeddings =========="
python tools/infer_reid_embeddings.py \
  "${QUERY_INPUT}" \
  --config-file "${CONFIG_FILE}" \
  --weights "${WEIGHTS_PATH}" \
  --output "${QUERY_OUTPUT}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --input-mode images

echo "========== Extract gallery embeddings =========="
python tools/infer_reid_embeddings.py \
  "${GALLERY_INPUT}" \
  --config-file "${CONFIG_FILE}" \
  --weights "${WEIGHTS_PATH}" \
  --output "${GALLERY_OUTPUT}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --input-mode images

echo "================================================"
echo "FastReID eval embeddings written:"
echo "  Query:   ${QUERY_OUTPUT}"
echo "  Gallery: ${GALLERY_OUTPUT}"
echo "Finished at: $(date)"
echo "================================================"
