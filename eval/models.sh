#!/usr/bin/env bash
# The same evaluation for several models served by Ollama, one after another - to find the smallest model that still
# reads a document well, and to check every model is given the window it should be (docs/READING_MODES.md, "Smaller
# models"). For each model the log records the window docuzent asked for and what Ollama actually loaded (ollama ps).
#
#   eval/models.sh <book.txt> <set name> <runs> <model> [model...]
#   eval/models.sh ~/books/gita.txt gita rag,rag-expanded,rag-kv,kv qwen3:0.6b llama3.2:1b qwen2.5:3b
#
# Ollama must be reachable from containers (EVAL_OLLAMA_URL, default the host's), and nothing else should hold the GPU:
# stop the llama.cpp server first. A model whose window is too small is refused before anything is read.
set -uo pipefail
cd "$(dirname "$0")/.."
BOOK="$1"; SET="$2"; RUNS="$3"; shift 3
OLLAMA="${EVAL_OLLAMA_URL:-http://host.docker.internal:11434}"
mkdir -p eval/results/logs
for m in "$@"; do
  log="eval/results/logs/$SET-$(echo "$m" | tr -c 'a-zA-Z0-9.\n' '-')-$(date +%Y%m%d-%H%M%S).log"
  echo "== $(date) $SET $m $RUNS" | tee -a eval/results/logs/models.log
  # what Ollama has loaded, while the evaluation runs (its CONTEXT column is the window it really gave the model)
  ( sleep 90; while docker ps -q --filter label=docuzent-eval=1 | grep -q .; do ollama ps 2>/dev/null | grep -F "$m" | sed "s/^/[ollama ps] /" && break; sleep 30; done ) >> "$log" 2>&1 &
  bash eval/run.sh "$BOOK" "$SET" "$RUNS" --ollama-url "$OLLAMA" --ollama-model "$m" >> "$log" 2>&1
  echo "== $(date) done ($?) -> $log" | tee -a eval/results/logs/models.log
  wait
  ollama stop "$m" >/dev/null 2>&1 || true
done
