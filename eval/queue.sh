#!/usr/bin/env bash
# Runs several evaluations one after another (one model server: never two at once). Each line of the queue file is
# `<book file> <set name> [runs]`; a line is removed once it has run. Logs go to eval/results/logs/.
#   eval/queue.sh eval/queue.txt
set -uo pipefail
cd "$(dirname "$0")/.."
QUEUE="$1"
mkdir -p eval/results/logs
# wait for any evaluation already running
busy() { [ -n "$(docker ps -q --filter label=docuzent-eval=1)" ] || [ -n "$(docker ps -q --filter ancestor=rust:1-bookworm)" ]; }
while busy; do sleep 20; done
while [ -s "$QUEUE" ]; do
  line="$(head -n1 "$QUEUE")"
  tail -n +2 "$QUEUE" > "$QUEUE.rest" && mv "$QUEUE.rest" "$QUEUE"
  [ -z "${line// }" ] && continue
  set -- $line
  log="eval/results/logs/$2-$(date +%Y%m%d-%H%M%S).log"
  echo "== $(date) $line" | tee -a eval/results/logs/queue.log
  bash eval/run.sh "$1" "$2" "${3:-rag,rag:guided,rag-expanded,rag-kv,rag-kv:expanded,kv}" > "$log" 2>&1
  echo "== $(date) done ($?) -> $log" | tee -a eval/results/logs/queue.log
done
