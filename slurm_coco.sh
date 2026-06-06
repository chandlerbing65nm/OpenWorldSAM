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
#SBATCH --output=logs/a-coco/output_%j.txt

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
MODEL_WEIGHTS=${MODEL_WEIGHTS:-checkpoints/model_final.pth}
VISION_PRETRAINED=${VISION_PRETRAINED:-checkpoints/sam2_hiera_large.pt}
OUTPUT_DIR=${OUTPUT_DIR:-./output_coco}
RUN_IDX=${RUN_IDX:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-0.0001}
MAX_ITER=${MAX_ITER:-10000}
NUM_WORKERS=${NUM_WORKERS:-4}
RANDOM_SUBSET_RATIO=${RANDOM_SUBSET_RATIO:-1.0}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
RUN_MODE=${RUN_MODE:-eval}
EVAL_TASK=${EVAL_TASK:-coco_instance}
COCO_INSTANCE_CONFIG=${COCO_INSTANCE_CONFIG:-configs/coco/instance-segmentation/Open-World-SAM2-CrossAttention.yaml}
COCO_INSTANCE_TEST_DATASET=${COCO_INSTANCE_TEST_DATASET:-coco_2017_val}
COCO_PANOPTIC_CONFIG=${COCO_PANOPTIC_CONFIG:-configs/coco/panoptic-segmentation/Open-World-SAM2-CrossAttention.yaml}
COCO_PANOPTIC_TEST_DATASET=${COCO_PANOPTIC_TEST_DATASET:-coco_2017_val_panoptic_with_sem_seg}

cd "$PROJECT_ROOT"

if [ "$RUN_MODE" = "eval" ]; then
  if [ "$EVAL_TASK" = "coco_instance" ]; then
    python train_net.py \
      --config-file "$COCO_INSTANCE_CONFIG" \
      --num-gpus 1 \
      --eval-only \
      --run_idx "$RUN_IDX" \
      -b "$BATCH_SIZE" \
      --lr "$LR" \
      MODEL.WEIGHTS "$MODEL_WEIGHTS" \
      MODEL.OpenWorldSAM2.VISION_PRETRAINED "$VISION_PRETRAINED" \
      DATASETS.TEST "('$COCO_INSTANCE_TEST_DATASET',)" \
      OUTPUT_DIR "$OUTPUT_DIR" \
      INPUT.IMAGE_SIZE "$IMAGE_SIZE" \
      DATALOADER.NUM_WORKERS "$NUM_WORKERS" \
      DATALOADER.RANDOM_SUBSET_RATIO "$RANDOM_SUBSET_RATIO"
  elif [ "$EVAL_TASK" = "coco_panoptic" ]; then
    python train_net.py \
      --config-file "$COCO_PANOPTIC_CONFIG" \
      --num-gpus 1 \
      --eval-only \
      --run_idx "$RUN_IDX" \
      -b "$BATCH_SIZE" \
      --lr "$LR" \
      MODEL.WEIGHTS "$MODEL_WEIGHTS" \
      MODEL.OpenWorldSAM2.VISION_PRETRAINED "$VISION_PRETRAINED" \
      DATASETS.TEST "('$COCO_PANOPTIC_TEST_DATASET',)" \
      OUTPUT_DIR "$OUTPUT_DIR" \
      INPUT.IMAGE_SIZE "$IMAGE_SIZE" \
      DATALOADER.NUM_WORKERS "$NUM_WORKERS" \
      DATALOADER.RANDOM_SUBSET_RATIO "$RANDOM_SUBSET_RATIO"
  else
    echo "Unsupported EVAL_TASK='$EVAL_TASK'. Expected 'coco_instance' or 'coco_panoptic'." >&2
    exit 1
  fi
else
  echo "Unsupported RUN_MODE='$RUN_MODE'. Expected 'eval'." >&2
  exit 1
fi
