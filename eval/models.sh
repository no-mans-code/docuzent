#!/usr/bin/env bash
# The same evaluation for several models served by Ollama, one after another - to find the smallest model that still
# reads a document well, and to check every model is given the window it should be (docs/READING_MODES.md, "Smaller
# models"). For each model the log records the window docuzent asked for and what Ollama actually loaded (ollama ps).
#
#   eval/models.sh <book.txt> <set name> <runs> <model> [model...]
#   eval/models.sh ~/books/gita.txt gita rag,rag-expanded,rag-kv,kv qwen3:0.6b llama3.2:1b qwen2.5:3b
#   EVAL_EXTRA='--context 1024' eval/models.sh ~/books/gita.txt gita rag,kv qwen3:0.6b   (extra evaluator options)
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
  bash eval/run.sh "$BOOK" "$SET" "$RUNS" --ollama-url "$OLLAMA" --ollama-model "$m" ${EVAL_EXTRA:-} >> "$log" 2>&1 &
  run=$!
  # what Ollama has loaded while this evaluation runs (its CONTEXT column is the window it really gave the model)
  while kill -0 "$run" 2>/dev/null; do
    if p="$(ollama ps 2>/dev/null | grep -F "$m ")"; then echo "[ollama ps] $p" >> "$log"; break; fi
    sleep 20
  done
  wait "$run"; rc=$?
  echo "== $(date) done ($rc) -> $log" | tee -a eval/results/logs/models.log
  ollama stop "$m" >/dev/null 2>&1 || true
done
