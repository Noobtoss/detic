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
ROOT=/nfs/scratch/staff/schmittth/code_hub/detic
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/detic_${SLURM_JOB_ID}_XXXXXX")

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
module load cuda/cuda-11.8.0

eval "$(conda shell.bash hook)"
conda activate conda-detic

export PYTHONPATH="$ROOT/custom/src:$ROOT:$ROOT/third_party/CenterNet2${PYTHONPATH:+:$PYTHONPATH}"

# ----- GET ARGS ----------------------------------------------------
EXP_MAP_FILE="$ROOT/custom/slurm/experiments_map.txt"
EXP_ARGS_FILE="$ROOT/custom/slurm/experiments.json"

EXP_NAME=$(grep -v '^[[:space:]]*#' "$EXP_MAP_FILE" | awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print $2; exit}')
if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no exp_name for task id $SLURM_ARRAY_TASK_ID in $EXP_MAP_FILE" >&2
    exit 1
fi

ARGS=$(python -c '
import json, sys
exp = json.load(open(sys.argv[1]))[sys.argv[2]]
print(" ".join(f"{k} {v}" for k, v in exp.items()))
' "$EXP_ARGS_FILE" "$EXP_NAME") || { echo "ERROR: exp_name '$EXP_NAME' not found in $EXP_ARGS_FILE" >&2; exit 1; }
ARGS="exp_name $EXP_NAME $ARGS"

declare -A KV
read -r -a ARR <<< "$ARGS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    KV["${ARR[$i]}"]="${ARR[$i+1]}"
done

RUN_NAME="${KV[exp_name]:-unnamed_experiment}"
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="$ROOT/runs/$RUN_NAME"
MODEL_CFG="$ROOT/${KV[config]}"
CKPT="$ROOT/${KV[ckpt]}"
DATA_DIR="$ROOT/${KV[data_dir]}"

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python "$ROOT/custom/src/train.py" \
       --run_name "$RUN_NAME" \
       --work_dir "$OUT_DIR" \
       --model_cfg "$MODEL_CFG" \
       --ckpt "$CKPT" \
       --data_dir "$DATA_DIR" \
       --ann_train "${KV[ann_train]}" \
       --ann_test "${KV[ann_test]}" \
       --workers "${SLURM_CPUS_PER_TASK:-1}"

# ----- CLEANUP -----------------------------------------------------
[[ -f "$OUT_DIR/model_final.pth" ]] && mv -f "$OUT_DIR/model_final.pth" "$OUT_DIR/${RUN_NAME}.pth"
KEEP_FILES=("${RUN_NAME}.pth")

for i in 1 2 3 4 5; do
    wandb sync --sync-all && break
    echo "sync attempt $i failed, retrying..."
    sleep 20
done

rm -rf "$TMPDIR"
rm -rf "$OUT_DIR/wandb"
find "$OUT_DIR" -type f $(printf ' ! -name %s' "${KEEP_FILES[@]}") -delete
find "$OUT_DIR" -type d -empty -delete
