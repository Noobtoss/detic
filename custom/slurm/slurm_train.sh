#!/bin/bash
#SBATCH --job-name=detic # Kurzname des Jobs
#SBATCH --array=1%1
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=p6             # p4
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G          # RAM pro CPU Kern #20G #32G #64G

# ----- DIRS --------------------------------------------------------
ROOT_DIR=/nfs/scratch/staff/schmittth/code_nexus/detic
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/detic_${SLURM_JOB_ID}_XXXXXX")

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$ROOT_DIR/custom/slurm/slurm_params.txt"
PARAMS=$(grep -v '^[[:space:]]*#' "$PARAMS_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")

declare -A KV
read -r -a ARR <<< "$PARAMS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    KV["${ARR[$i]}"]="${ARR[$i+1]}"
done

RUN_NAME="${KV[exp_name]:-unnamed_experiment}"
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="$ROOT_DIR/runs/$RUN_NAME"
MODEL_CFG="$ROOT_DIR/${KV[config]}"
WEIGHTS="$ROOT_DIR/${KV[weights]}"
DATASET_DIR="$ROOT_DIR/datasets/03Detic"

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
module load cuda/cuda-11.8.0

eval "$(conda shell.bash hook)"
conda activate conda-detic

export PYTHONPATH="$ROOT_DIR/custom/src:$ROOT_DIR:$ROOT_DIR/third_party/CenterNet2${PYTHONPATH:+:$PYTHONPATH}"

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python "$ROOT_DIR/custom/src/train.py" \
       --run_name "$RUN_NAME" \
       --work_dir "$OUT_DIR" \
       --model_cfg "$MODEL_CFG" \
       --weights "$WEIGHTS" \
       --train_json "$DATASET_DIR/annotations/annotation_train.json" \
       --test_json "$DATASET_DIR/annotations/annotation_test.json" \
       --imgs_dir "$DATASET_DIR/Images" \
       --workers "${SLURM_CPUS_PER_TASK:-1}"

# ----- CLEANUP -----------------------------------------------------
# KEEP_FILES=("metrics.csv" "results.csv" "last.pt")

# for i in 1 2 3 4 5; do
#     wandb sync --sync-all && break
#     echo "sync attempt $i failed, retrying..."
#     sleep 20
# done

# rm -rf "$TMPDIR"
# rm -rf "$OUT_DIR/wandb"
# find "$OUT_DIR" -type f $(printf ' ! -name %s' "${KEEP_FILES[@]}") -delete
# find "$OUT_DIR" -type d -empty -delete
