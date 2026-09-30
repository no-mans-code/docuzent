#!/usr/bin/env bash
# Runs docuzent-eval inside a container next to a llama.cpp server, sharing its KV directory (through a scoped store:
# the evaluation can never touch another app's saved files) - the way the numbers in docs/READING_MODES.md were made.
#
#   eval/run.sh <book.txt> <set name> [runs] [extra docuzent-eval args...]
#   eval/run.sh ~/books/gita.txt gita
#   eval/run.sh ~/books/harry-potter.txt harry-potter rag,rag-kv,kv --only 1,2,3
#
# Environment: EVAL_NETWORK (the model server's Docker network, default thebook_default), EVAL_KV_VOLUME (its KV
# volume, default thebook_thebook-kv), EVAL_LLM_URL (default http://llm:8080), EVAL_EMBED_URL (default the host's
# Ollama). Books are not in this repository: bring your own copy.
set -euo pipefail
cd "$(dirname "$0")/.."
BOOK="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
SET="$2"
RUNS="${3:-rag,rag:guided,rag-expanded,rag-kv,rag-kv:expanded,kv}"
shift 3 2>/dev/null || shift $#
docker run --rm \
  --network "${EVAL_NETWORK:-thebook_default}" \
  --add-host host.docker.internal:host-gateway \
  -v "${EVAL_KV_VOLUME:-thebook_thebook-kv}:/kv" \
  -v "$PWD:/src" -v "$(dirname "$BOOK"):/books:ro" \
  -v docuzent-eval-target:/target -v docuzent-cargo-registry:/usr/local/cargo/registry \
  -e CARGO_TARGET_DIR=/target -w /src rust:1-bookworm \
  bash -c "cargo build -q --release -p docuzent-eval && /target/release/docuzent-eval \
    --book '/books/$(basename "$BOOK")' --questions 'eval/sets/$SET.json' \
    --llm-url '${EVAL_LLM_URL:-http://llm:8080}' --kv-dir /kv \
    --embed-url '${EVAL_EMBED_URL:-http://host.docker.internal:11434}' \
    --runs '$RUNS' --work eval/work --out eval/results $*"
