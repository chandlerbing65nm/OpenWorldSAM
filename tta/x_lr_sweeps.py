

# ------
# ------ source

SEED=1 RUN_IDX=0 TTA_METHOD=source CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_source.yaml TTA_MODE=cont_tta \
TTA_METHOD=source OUTPUT_DIR=./tta/output_dutuseg_c/source TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=0 TTA_METHOD=source CONFIG_FILE=tta/cfgs/suim_c/suim_c_source.yaml TTA_MODE=cont_tta \
TTA_METHOD=source OUTPUT_DIR=./tta/output_suim_c/source TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

# -------
# ------- tent

SEED=1 RUN_IDX=1 TTA_METHOD=tent CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_tent.yaml TTA_MODE=cont_tta \
TTA_METHOD=tent OUTPUT_DIR=./tta/output_dutuseg_c/tent TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=tent CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_tent.yaml TTA_MODE=cont_tta \
TTA_METHOD=tent OUTPUT_DIR=./tta/output_dutuseg_c/tent TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=tent CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_tent.yaml TTA_MODE=cont_tta \
TTA_METHOD=tent OUTPUT_DIR=./tta/output_dutuseg_c/tent TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

# -------

SEED=1 RUN_IDX=1 TTA_METHOD=tent CONFIG_FILE=tta/cfgs/suim_c/suim_c_tent.yaml TTA_MODE=cont_tta \
TTA_METHOD=tent OUTPUT_DIR=./tta/output_suim_c/tent TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=tent CONFIG_FILE=tta/cfgs/suim_c/suim_c_tent.yaml TTA_MODE=cont_tta \
TTA_METHOD=tent OUTPUT_DIR=./tta/output_suim_c/tent TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=tent CONFIG_FILE=tta/cfgs/suim_c/suim_c_tent.yaml TTA_MODE=cont_tta \
TTA_METHOD=tent OUTPUT_DIR=./tta/output_suim_c/tent TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

# ------
# ------ cotta

SEED=1 RUN_IDX=1 TTA_METHOD=cotta CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_cotta.yaml TTA_MODE=cont_tta \
TTA_METHOD=cotta OUTPUT_DIR=./tta/output_dutuseg_c/cotta TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=cotta CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_cotta.yaml TTA_MODE=cont_tta \
TTA_METHOD=cotta OUTPUT_DIR=./tta/output_dutuseg_c/cotta TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=cotta CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_cotta.yaml TTA_MODE=cont_tta \
TTA_METHOD=cotta OUTPUT_DIR=./tta/output_dutuseg_c/cotta TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

# -------

SEED=1 RUN_IDX=1 TTA_METHOD=cotta CONFIG_FILE=tta/cfgs/suim_c/suim_c_cotta.yaml TTA_MODE=cont_tta \
TTA_METHOD=cotta OUTPUT_DIR=./tta/output_suim_c/cotta TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=cotta CONFIG_FILE=tta/cfgs/suim_c/suim_c_cotta.yaml TTA_MODE=cont_tta \
TTA_METHOD=cotta OUTPUT_DIR=./tta/output_suim_c/cotta TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=cotta CONFIG_FILE=tta/cfgs/suim_c/suim_c_cotta.yaml TTA_MODE=cont_tta \
TTA_METHOD=cotta OUTPUT_DIR=./tta/output_suim_c/cotta TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

# ------
# ------ sar

SEED=1 RUN_IDX=1 TTA_METHOD=sar CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_sar.yaml TTA_MODE=cont_tta \
TTA_METHOD=sar OUTPUT_DIR=./tta/output_dutuseg_c/sar TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=sar CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_sar.yaml TTA_MODE=cont_tta \
TTA_METHOD=sar OUTPUT_DIR=./tta/output_dutuseg_c/sar TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=sar CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_sar.yaml TTA_MODE=cont_tta \
TTA_METHOD=sar OUTPUT_DIR=./tta/output_dutuseg_c/sar TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

# -------

SEED=1 RUN_IDX=1 TTA_METHOD=sar CONFIG_FILE=tta/cfgs/suim_c/suim_c_sar.yaml TTA_MODE=cont_tta \
TTA_METHOD=sar OUTPUT_DIR=./tta/output_suim_c/sar TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=sar CONFIG_FILE=tta/cfgs/suim_c/suim_c_sar.yaml TTA_MODE=cont_tta \
TTA_METHOD=sar OUTPUT_DIR=./tta/output_suim_c/sar TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=sar CONFIG_FILE=tta/cfgs/suim_c/suim_c_sar.yaml TTA_MODE=cont_tta \
TTA_METHOD=sar OUTPUT_DIR=./tta/output_suim_c/sar TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

# ------
# ------ rdumb

SEED=1 RUN_IDX=1 TTA_METHOD=rdumb CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_rdumb.yaml TTA_MODE=cont_tta \
TTA_METHOD=rdumb OUTPUT_DIR=./tta/output_dutuseg_c/rdumb TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=rdumb CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_rdumb.yaml TTA_MODE=cont_tta \
TTA_METHOD=rdumb OUTPUT_DIR=./tta/output_dutuseg_c/rdumb TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=rdumb CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_rdumb.yaml TTA_MODE=cont_tta \
TTA_METHOD=rdumb OUTPUT_DIR=./tta/output_dutuseg_c/rdumb TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

# -------

SEED=1 RUN_IDX=1 TTA_METHOD=rdumb CONFIG_FILE=tta/cfgs/suim_c/suim_c_rdumb.yaml TTA_MODE=cont_tta \
TTA_METHOD=rdumb OUTPUT_DIR=./tta/output_suim_c/rdumb TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=rdumb CONFIG_FILE=tta/cfgs/suim_c/suim_c_rdumb.yaml TTA_MODE=cont_tta \
TTA_METHOD=rdumb OUTPUT_DIR=./tta/output_suim_c/rdumb TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=rdumb CONFIG_FILE=tta/cfgs/suim_c/suim_c_rdumb.yaml TTA_MODE=cont_tta \
TTA_METHOD=rdumb OUTPUT_DIR=./tta/output_suim_c/rdumb TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

# ------
# ------ deyo

SEED=1 RUN_IDX=1 TTA_METHOD=deyo CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_deyo.yaml TTA_MODE=cont_tta \
TTA_METHOD=deyo OUTPUT_DIR=./tta/output_dutuseg_c/deyo TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=deyo CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_deyo.yaml TTA_MODE=cont_tta \
TTA_METHOD=deyo OUTPUT_DIR=./tta/output_dutuseg_c/deyo TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=deyo CONFIG_FILE=tta/cfgs/dutuseg_c/dutuseg_c_deyo.yaml TTA_MODE=cont_tta \
TTA_METHOD=deyo OUTPUT_DIR=./tta/output_dutuseg_c/deyo TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_dutuseg_tta.sh

# -------

SEED=1 RUN_IDX=1 TTA_METHOD=deyo CONFIG_FILE=tta/cfgs/suim_c/suim_c_deyo.yaml TTA_MODE=cont_tta \
TTA_METHOD=deyo OUTPUT_DIR=./tta/output_suim_c/deyo TTA_LR=1e-3 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=2 TTA_METHOD=deyo CONFIG_FILE=tta/cfgs/suim_c/suim_c_deyo.yaml TTA_MODE=cont_tta \
TTA_METHOD=deyo OUTPUT_DIR=./tta/output_suim_c/deyo TTA_LR=1e-4 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh

SEED=1 RUN_IDX=3 TTA_METHOD=deyo CONFIG_FILE=tta/cfgs/suim_c/suim_c_deyo.yaml TTA_MODE=cont_tta \
TTA_METHOD=deyo OUTPUT_DIR=./tta/output_suim_c/deyo TTA_LR=1e-5 TTA_BATCH_SIZE=1 \
sbatch ./slurm_suim_tta.sh