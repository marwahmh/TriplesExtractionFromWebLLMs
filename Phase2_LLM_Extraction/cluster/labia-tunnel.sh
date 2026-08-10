#!/usr/bin/env bash
# Run on your Mac. Forwards a local port to a vLLM server running on a Lab-IA compute node.
# Leave it running in its own terminal; Ctrl-C closes the tunnel.
#
#   ./labia-tunnel.sh                 # one server running -> just connect
#   ./labia-tunnel.sh n5              # pick by node when several are up
#   ./labia-tunnel.sh 961650          # ...or by job id
#   PORT=8001 ./labia-tunnel.sh n5    # second tunnel alongside the first
#
set -euo pipefail

TARGET="${1:-}"
PORT="${PORT:-8000}"

# One round trip: job id, node, and the model each one serves (from its log header).
JOBS=$(ssh slurm '
  for j in $(squeue -u $USER -h -n vllm -t RUNNING -o %i); do
    n=$(squeue -h -j "$j" -o %N)
    m=$(grep -m1 "^ model" "vllm-$j.log" 2>/dev/null | sed "s/^ model *: *//")
    echo "$j $n ${m:-unknown}"
  done')

if [ -z "$JOBS" ]; then
  echo "No RUNNING job named 'vllm'. Check with:  ssh slurm 'squeue -u \$USER'"
  exit 1
fi

COUNT=$(printf '%s\n' "$JOBS" | wc -l | tr -d ' ')

if [ -n "$TARGET" ]; then
  NODE=$(printf '%s\n' "$JOBS" | awk -v t="$TARGET" '$1==t || $2==t {print $2; exit}')
  if [ -z "$NODE" ]; then
    echo "No running vLLM job matches '$TARGET'. Running now:"
    printf '  %s\n' "$JOBS"
    exit 1
  fi
elif [ "$COUNT" -eq 1 ]; then
  NODE=$(printf '%s\n' "$JOBS" | awk '{print $2}')
else
  echo "Several vLLM servers are running — say which one:"
  printf '%s\n' "$JOBS" | while read -r id node model; do
    printf '  %-28s # job %s, %s\n' "$0 $node" "$id" "$model"
  done
  echo
  echo "Use PORT=8001 (etc.) to open more than one tunnel at a time."
  exit 1
fi

MODEL=$(printf '%s\n' "$JOBS" | awk -v n="$NODE" '$2==n {$1=""; $2=""; print substr($0,3); exit}')
echo "tunnel: localhost:${PORT}  ->  ${NODE}:8000   [${MODEL}]   (Ctrl-C to stop)"
exec ssh -N -L "${PORT}:${NODE}.lab-ia.local:8000" slurm
