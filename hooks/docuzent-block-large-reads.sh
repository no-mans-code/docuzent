#!/usr/bin/env bash
# coding agent PreToolUse hook: intercepts a `Read` call on a file above a
# size threshold and redirects the agent to docuzent-mcp's `ask_document`
# tool instead - so the token savings from delegating large reads to a
# local model don't depend on the agent remembering that tool exists,
# every session. Inspired by token_save_mcp's PreToolUse hook (see
# https://github.com/no-mans-code/docuzent/issues/36).
#
# This file is a template to copy, not something active just by living
# in this repo: copy it into your own project's `.assistant/hooks/`
# directory and wire it into `.assistant/settings.json` - see
# `hooks/settings.snippet.json` in this repo and the README's
# "Auto-enforcing hook" section for the exact steps.
#
# Requires `jq`. Deliberately short and readable in one sitting - this
# runs on every Read call, so it should be auditable at a glance.
set -uo pipefail
# Deliberately not `-e`: malformed/unexpected stdin must fall through to
# "exit 0, do nothing" rather than abort with a jq error - a hook that
# can crash the tool call it's watching is worse than one that
# occasionally misses a large read.

# How large a file has to be before this redirects to ask_document -
# override per project via the environment (e.g. in
# .assistant/settings.json's hook `env`, or your own shell profile).
THRESHOLD_BYTES="${DOCUZENT_HOOK_SIZE_THRESHOLD_BYTES:-8000}"

input="$(cat)"
tool_name="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)" || exit 0
if [ "$tool_name" != "Read" ]; then
  exit 0 # not a Read call - nothing for this hook to do
fi

file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)" || exit 0
if [ -z "$file_path" ] || [ ! -f "$file_path" ]; then
  exit 0 # no path, or it doesn't exist - let the real Read report that error itself
fi

# GNU stat (Linux/git-bash) then BSD stat (macOS) - whichever this
# platform has.
size_bytes="$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null || echo 0)"
if [ "$size_bytes" -le "$THRESHOLD_BYTES" ]; then
  exit 0 # small enough that reading it directly is fine
fi

reason="This file is ${size_bytes} bytes (over the ${THRESHOLD_BYTES}-byte threshold) - reading it directly spends a large chunk of your own context on raw file bytes. Use the docuzent MCP server instead: ask_document(paths=[\"${file_path}\"], question=\"<a real, specific question about this file>\"). It delegates the read to a local model and returns only the answer, not the file's full contents."

jq -n --arg reason "$reason" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $reason}}'
exit 0
