#!/bin/sh
# Ambient AI - tmux focus hook (Phase 2 Unit 8)
#
# Invoked by tmux on pane-focus-in/pane-focus-out/window-focused events.
# Writes a single JSONL line to ~/.ambient/focus-events.jsonl.
#
# Privacy contract (cites docs/PRIVACY.md clauses 6, 7):
# - Captured fields: hook event name, pane id, window index, session name, ts.
# - NEVER captured: pane_current_command, pane_current_path, pane_title,
#   pane_current_pid. tmux exposes these via #{...} format strings; this hook
#   intentionally references none of them.
# - Off by default; installed and removed by `ambient tmux-focus-enable` /
#   `ambient tmux-focus-disable`.
#
# Marker comment for idempotent install/remove: # ambient-managed
#
# Usage (set automatically by tmux_focus.py):
#   tmux set-hook -g pane-focus-in 'run-shell "/path/ambient-focus-hook.sh pane-focus-in"  # ambient-managed'

set -eu

EVENT_NAME="${1:-unknown}"
EVENTS_PATH="${AMBIENT_FOCUS_EVENTS_PATH:-$HOME/.ambient/focus-events.jsonl}"

# Resolve fields from the tmux environment. tmux exports $TMUX_PANE
# unconditionally inside `run-shell`. session name + window index come via
# `tmux display-message -p`, which we call only with structural format
# strings (#S session name, #I window index). Window-title and
# current-command/path interpolations are deliberately not used here.
PANE_ID="${TMUX_PANE:-}"
SESSION_NAME=""
WINDOW_INDEX=""
if command -v tmux >/dev/null 2>&1 && [ -n "${TMUX:-}" ]; then
    SESSION_NAME="$(tmux display-message -p '#S' 2>/dev/null || echo "")"
    WINDOW_INDEX="$(tmux display-message -p '#I' 2>/dev/null || echo "")"
fi

# ISO 8601 UTC timestamp, second precision (sufficient for focus events).
TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

mkdir -p "$(dirname "$EVENTS_PATH")"

# Build JSONL line. Field set is fixed: never add anything sourced from
# pane_title, pane_current_command, or pane_current_path here.
printf '{"ts":"%s","source":"tmux","event":"%s","pane_id":"%s","window_index":"%s","session_name":"%s"}\n' \
    "$TS" \
    "$EVENT_NAME" \
    "$PANE_ID" \
    "$WINDOW_INDEX" \
    "$SESSION_NAME" \
    >> "$EVENTS_PATH"
