#!/bin/bash

# Configuration Variables
NUM_GPU=4
INSTANCES=40
NUM_EPISODES_PER_INSTANCE=25
MAX_STEPS_PER_EPISODE=500
TASK="ObjectNav"
CFG="ObjectNav"
NAME="xxx"
SLEEP_INTERVAL=200
LOG_FREQ=200
PORT=1000
VENV_NAME="vlm_nav" # Name of the conda environment
CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT}"

# === Visualization (added) ===
ENABLE_VIS=1              
BAR_WIDTH=50              
draw_bar() {
  local done=$1
  local total=$2
  local width=${BAR_WIDTH:-50}
  local filled=$(( done * width / total ))
  local empty=$(( width - filled ))

  # 构造条形
  local bar_filled
  local bar_empty
  bar_filled=$(printf "%0.s#" $(seq 1 $filled))
  bar_empty=$(printf "%0.s-" $(seq 1 $empty))

  printf "\r[%s%s] %d/%d  %s" "$bar_filled" "$bar_empty" "$done" "$total" "$(date '+%H:%M:%S')"
}

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="aggregator_${NAME}"

# Start Aggregator Session
tmux new-session -d -s "$AGGREGATOR_SESSION" \
  "bash -i -c 'conda activate ${VENV_NAME} && python scripts/aggregator.py --name ${TASK}_${NAME} --sleep ${SLEEP_INTERVAL} --port ${PORT}'"
SESSION_NAMES+=("$AGGREGATOR_SESSION")

# Cleanup Function
cleanup() {
  echo "\nCaught interrupt signal. Cleaning up tmux sessions..."

  for session in "${SESSION_NAMES[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      echo "Killed session: $session"
    fi
  done

}

# Trap SIGINT to Run Cleanup
trap cleanup SIGINT

# Start Tmux Sessions for Each Instance
for instance_id in $(seq 0 $((INSTANCES - 1))); do
  GPU_ID=$((instance_id % NUM_GPU))
  SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"

  tmux new-session -d -s "$SESSION_NAME" \
    "bash -i -c 'conda activate ${VENV_NAME} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id'"
  SESSION_NAMES+=("$SESSION_NAME")
done

# Monitor Tmux Sessions
while true; do
  sleep $SLEEP_INTERVAL

  ALL_DONE=true
  DONE_COUNT=0   # === Visualization (added)

  for instance_id in $(seq 0 $((INSTANCES - 1))); do
    SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "$SESSION_NAME finished"
      DONE_COUNT=$((DONE_COUNT + 1))   # === Visualization (added)
    else
      ALL_DONE=false
    fi
  done

  # === Visualization (added)
  if [ "$ENABLE_VIS" -eq 1 ]; then
    draw_bar "$DONE_COUNT" "$INSTANCES"
  fi

  if $ALL_DONE; then
    # === Visualization (added) 
    if [ "$ENABLE_VIS" -eq 1 ]; then
      printf "\n"
    fi

    echo "DONE"
    echo "$(date): Sending termination signal to aggregator."
    curl -X POST http://localhost:${port}/terminate
    if [ $? -eq 0 ]; then
      echo "$(date): Termination signal sent successfully."
    else
      echo "$(date): Failed to send termination signal."
    fi

    sleep 10
    if tmux has-session -t "$AGGREGATOR_SESSION" 2>/dev/null; then
      tmux kill-session -t "$AGGREGATOR_SESSION"
      echo "Killed session: $AGGREGATOR_SESSION"
    fi
    break
  fi

done