#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --account=project_465002853
#SBATCH --output=logs/a-refcocog/output_%j.txt

# Use node-local scratch for MIOpen DB (avoid Lustre/NFS locking issues)
MIOPEN_LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER}/miopen-${SLURM_JOB_ID}"
export MIOPEN_USER_DB_PATH="$MIOPEN_LOCAL"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_LOCAL"
mkdir -p "$MIOPEN_LOCAL"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=1

# Activate conda in non-interactive shells and activate the env
source /scratch/project_465002853/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/project_465002853/miniconda3/envs/detectron2

# Hugging Face cache in a directory you own
export HF_HOME="/scratch/project_465002853/hf_cache_${USER}"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_TELEMETRY=1

export DETECTRON2_DATASETS=/scratch/project_465002853/datasets

PROJECT_ROOT=/flash/project_465002853/projects/openworldsam
CONFIG_FILE=${CONFIG_FILE:-configs/refcoco/Open-World-SAM2-CrossAttention.yaml}
MODEL_WEIGHTS=${MODEL_WEIGHTS:-""}
VISION_PRETRAINED=${VISION_PRETRAINED:-checkpoints/sam2_hiera_large.pt}
OUTPUT_DIR=${OUTPUT_DIR:-./output_refcocog}
RUN_IDX=${RUN_IDX:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.0001}
MAX_ITER=${MAX_ITER:-100000}
EVAL_PERIOD=${EVAL_PERIOD:-5000}
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-5000}
NUM_WORKERS=${NUM_WORKERS:-4}
RANDOM_SUBSET_RATIO=${RANDOM_SUBSET_RATIO:-1.0}
IMAGE_SIZE=${IMAGE_SIZE:-1024}

mkdir -p "$PROJECT_ROOT/logs/a-refcocog"
cd "$PROJECT_ROOT"


# # finetuning
# python train_net.py \
#   --config-file "$CONFIG_FILE" \
#   --num-gpus 1 \
#   --run_idx "$RUN_IDX" \
#   -b "$BATCH_SIZE" \
#   --lr "$LR" \
#   MODEL.WEIGHTS "$MODEL_WEIGHTS" \
#   MODEL.OpenWorldSAM2.VISION_PRETRAINED "$VISION_PRETRAINED" \
#   OUTPUT_DIR "$OUTPUT_DIR" \
#   SOLVER.MAX_ITER "$MAX_ITER" \
#   SOLVER.CHECKPOINT_PERIOD "$CHECKPOINT_PERIOD" \
#   TEST.EVAL_PERIOD "$EVAL_PERIOD" \
#   DATALOADER.NUM_WORKERS "$NUM_WORKERS" \
#   DATALOADER.RANDOM_SUBSET_RATIO "$RANDOM_SUBSET_RATIO" \
#   INPUT.IMAGE_SIZE "$IMAGE_SIZE"

# evaluation
python train_net.py \
  --config-file "$CONFIG_FILE" \
  --num-gpus 1 \
  --eval-only \
  --run_idx "$RUN_IDX" \
  -b "$BATCH_SIZE" \
  --lr "$LR" \
  MODEL.WEIGHTS "./checkpoints/refcocog_model_final.pth" \
  MODEL.OpenWorldSAM2.VISION_PRETRAINED "$VISION_PRETRAINED" \
  OUTPUT_DIR "$OUTPUT_DIR" \
  DATALOADER.NUM_WORKERS "$NUM_WORKERS" \
  DATALOADER.RANDOM_SUBSET_RATIO "$RANDOM_SUBSET_RATIO" \
  INPUT.IMAGE_SIZE "$IMAGE_SIZE"