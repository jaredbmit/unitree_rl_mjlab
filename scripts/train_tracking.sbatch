#!/bin/bash
# Tracking training (task Unitree-G1-Tracking-No-State-Estimation) on one GPU, with resume.
#
#   sbatch -J <job name> scripts/train_tracking.sbatch <run name> <motion.npz> [num_envs=4096] [total_iters=30001]
#
# Submit from the repo root. Without Slurm, run it with bash; the command it runs is
#   .venv/bin/python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion-file <motion.npz> \
#     --env.scene.num-envs 4096 --agent.logger tensorboard --agent.run-name <run name> --agent.max-iterations 30001
# (from the output dir; train.py writes logs/rsl_rl/g1_tracking/<date>_<run name>/ relative to the cwd).
#
# Output dir: $OUT (default: logs/ in the repo; gitignored). Checkpoints model_<it>.pt every 500 iterations; each
# save also writes policy.onnx and <run dir>.onnx next to them.
# Resume: resubmitting the same command (or a Slurm requeue) finds the newest model_*.pt of <run name> and continues
# from it in a new <date>_<run name> dir until total_iters.
#
# Edit for your cluster: partition, account, qos.
##SBATCH --partition=<partition>
##SBATCH --account=<account>
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=trk
#SBATCH --output=slurm-%x_%j.out
set -euo pipefail
RUN="${1:?run name}"
MOTION="$(realpath "${2:?motion npz}")"
NUM_ENVS="${3:-4096}"
TOTAL="${4:-30001}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO/scripts/train.py" ] || REPO="$SLURM_SUBMIT_DIR"  # sbatch runs a spooled copy of this script
OUT="$(realpath -m "${OUT:-$REPO}")"
EXP=$OUT/logs/rsl_rl/g1_tracking
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled MUJOCO_GL=egl
mkdir -p "$EXP"
cd "$OUT"
echo "run $RUN motion $MOTION envs $NUM_ENVS total $TOTAL | $(hostname) job ${SLURM_JOB_ID:-none} $(date -Is) | repo $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

RESUME=()
DONE=0
LAST=$(ls -1 "$EXP"/*_"$RUN"/model_*.pt 2>/dev/null | awk -F'model_|\\.pt' '{print $(NF-1), $0}' | sort -n | tail -1 | cut -d' ' -f2- || true)
if [ -n "$LAST" ]; then
  DONE=$(basename "$LAST" .pt | sed 's/model_//')
  RESUME=(--agent.resume True --agent.load-run "$(basename "$(dirname "$LAST")")" --agent.load-checkpoint "$(basename "$LAST")")
  echo "resuming from $LAST (iteration $DONE)"
fi
LEFT=$(( TOTAL - DONE ))
if [ "$LEFT" -le 1 ]; then echo "already at $DONE / $TOTAL, nothing to do"; exit 0; fi
echo "${SLURM_JOB_ID:-none} $RUN from $DONE, $LEFT iterations, $(date -Is)" >> "$EXP/jobs.txt"

exec "$REPO/.venv/bin/python" "$REPO/scripts/train.py" Unitree-G1-Tracking-No-State-Estimation \
  --motion-file "$MOTION" --env.scene.num-envs "$NUM_ENVS" \
  --agent.logger tensorboard --agent.run-name "$RUN" --agent.max-iterations "$LEFT" "${RESUME[@]}"
