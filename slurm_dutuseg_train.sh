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
#SBATCH --output=logs/a-dutuseg/output_%j.txt

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

# ---------------------------------------------------------------------------
# Experiment / config selection
# ---------------------------------------------------------------------------
CONFIG_FILE=${CONFIG_FILE:-configs/pascal/semantic-segmentation/Open-World-SAM2-CrossAttention.yaml}
MODEL_WEIGHTS=${MODEL_WEIGHTS:-checkpoints/voc12_model_final.pth}
VISION_PRETRAINED=${VISION_PRETRAINED:-checkpoints/sam2_hiera_large.pt}
OUTPUT_DIR=${OUTPUT_DIR:-./output_dutuseg_semseg}
RUN_IDX=${RUN_IDX:-0}
RUN_MODE=${RUN_MODE:-train}

# ---------------------------------------------------------------------------
# Core training schedule (matches Base-COCO / Open-World-SAM2 configs)
# ---------------------------------------------------------------------------
# Match SOLVER.IMS_PER_BATCH from Base-COCO / Open-World-SAM2 config
BATCH_SIZE=${BATCH_SIZE:-1}
# Match SOLVER.BASE_LR from Base-COCO config
LR=${LR:-0.00001}
# Match SOLVER.MAX_ITER from Base-COCO / Open-World-SAM2 config
MAX_ITER=${MAX_ITER:-50000}

# ---------------------------------------------------------------------------
# Data loading / DUT-USEG semantic segmentation dataset
# ---------------------------------------------------------------------------
NUM_WORKERS=${NUM_WORKERS:-4}
RANDOM_SUBSET_RATIO=${RANDOM_SUBSET_RATIO:-1.0}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
DUTUSEG_DATASET_ROOT=${DUTUSEG_DATASET_ROOT:-/scratch/project_465002853/datasets/dut-useg/DUT-USEG}
DUTUSEG_MASKS_ROOT=${DUTUSEG_MASKS_ROOT:-${DUTUSEG_DATASET_ROOT}/SegmentationClassVisual}
DUTUSEG_NUM_CLASSES=${DUTUSEG_NUM_CLASSES:-5}

# ---------------------------------------------------------------------------
# Optimizer and LR schedule (from Base-COCO config)
# ---------------------------------------------------------------------------
OPTIMIZER=${OPTIMIZER:-ADAMW}
LR_SCHEDULER_NAME=${LR_SCHEDULER_NAME:-WarmupMultiStepLR}
# Match SOLVER.STEPS from Base-COCO config
SOLVER_STEPS=${SOLVER_STEPS:-"(20000, 30000)"}
# Match SOLVER.WEIGHT_DECAY from Base-COCO config
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
# Match SOLVER.BACKBONE_MULTIPLIER from Base-COCO config
BACKBONE_MULTIPLIER=${BACKBONE_MULTIPLIER:-0.1}
# Match SOLVER.WARMUP_FACTOR / WARMUP_ITERS from Base-COCO config
WARMUP_FACTOR=${WARMUP_FACTOR:-1.0}
WARMUP_ITERS=${WARMUP_ITERS:-10}
MOMENTUM=${MOMENTUM:-0.9}

# ---------------------------------------------------------------------------
# Instance post-processing thresholds for training-time eval
# ---------------------------------------------------------------------------
IOU_THRESHOLD=${IOU_THRESHOLD:-0.25}
NMS_THRESHOLD=${NMS_THRESHOLD:-0.7}
DETECTIONS_PER_IMAGE=${DETECTIONS_PER_IMAGE:-100}

# ---------------------------------------------------------------------------
# OpenWorldSAM-specific training knobs
# ---------------------------------------------------------------------------
TRAIN_VISUAL_ENCODER=${TRAIN_VISUAL_ENCODER:-True}
TRAIN_MASK_DECODER=${TRAIN_MASK_DECODER:-False}
TRAIN_PROMPT_ENCODER=${TRAIN_PROMPT_ENCODER:-False}
TRAIN_VLM=${TRAIN_VLM:-False}
# Match MODEL.OpenWorldSAM2.{MASK,DICE}_WEIGHT from Open-World-SAM2 config
DICE_WEIGHT=${DICE_WEIGHT:-0.0}
MASK_WEIGHT=${MASK_WEIGHT:-20.0}
NUM_OBJECT_QUERIES=${NUM_OBJECT_QUERIES:-20}

# ---------------------------------------------------------------------------
# Evaluation and checkpoint cadence
# ---------------------------------------------------------------------------
# Match TEST.EVAL_PERIOD from Base-COCO config
EVAL_PERIOD=${EVAL_PERIOD:-2000}
# Match SOLVER.CHECKPOINT_PERIOD from Open-World-SAM2 config
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-25000}

export DUTUSEG_DATASET_ROOT
export DUTUSEG_MASKS_ROOT

cd "$PROJECT_ROOT"

if [ "$RUN_MODE" = "train" ]; then
  python train_net.py \
    --config-file "$CONFIG_FILE" \
    --num-gpus 1 \
    --run_idx "$RUN_IDX" \
    -b "$BATCH_SIZE" \
    --lr "$LR" \
    MODEL.WEIGHTS "$MODEL_WEIGHTS" \
    MODEL.OpenWorldSAM2.VISION_PRETRAINED "$VISION_PRETRAINED" \
    DATASETS.TRAIN "('dutuseg_sem_seg_train',)" \
    DATASETS.TEST "('dutuseg_sem_seg_val',)" \
    OUTPUT_DIR "$OUTPUT_DIR" \
    INPUT.IMAGE_SIZE "$IMAGE_SIZE" \
    SOLVER.MAX_ITER "$MAX_ITER" \
    SOLVER.OPTIMIZER "$OPTIMIZER" \
    SOLVER.LR_SCHEDULER_NAME "$LR_SCHEDULER_NAME" \
    SOLVER.STEPS "$SOLVER_STEPS" \
    SOLVER.WEIGHT_DECAY "$WEIGHT_DECAY" \
    SOLVER.BACKBONE_MULTIPLIER "$BACKBONE_MULTIPLIER" \
    SOLVER.WARMUP_FACTOR "$WARMUP_FACTOR" \
    SOLVER.WARMUP_ITERS "$WARMUP_ITERS" \
    SOLVER.MOMENTUM "$MOMENTUM" \
    TEST.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.CHECKPOINT_PERIOD "$CHECKPOINT_PERIOD" \
    MODEL.SEM_SEG_HEAD.NUM_CLASSES "$DUTUSEG_NUM_CLASSES" \
    MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON "True" \
    MODEL.OpenWorldSAM2.TEST.INSTANCE_ON "False" \
    MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON "False" \
    MODEL.OpenWorldSAM2.TRAIN_VISUAL_ENCODER "$TRAIN_VISUAL_ENCODER" \
    MODEL.OpenWorldSAM2.TRAIN_MASK_DECODER "$TRAIN_MASK_DECODER" \
    MODEL.OpenWorldSAM2.TRAIN_PROMPT_ENCODER "$TRAIN_PROMPT_ENCODER" \
    MODEL.OpenWorldSAM2.TRAIN_VLM "$TRAIN_VLM" \
    MODEL.OpenWorldSAM2.DICE_WEIGHT "$DICE_WEIGHT" \
    MODEL.OpenWorldSAM2.MASK_WEIGHT "$MASK_WEIGHT" \
    MODEL.OpenWorldSAM2.NUM_OBJECT_QUERIES "$NUM_OBJECT_QUERIES" \
    DATALOADER.NUM_WORKERS "$NUM_WORKERS" \
    DATALOADER.RANDOM_SUBSET_RATIO "$RANDOM_SUBSET_RATIO"
fi