#!/bin/sh
# Ambient AI - tmux focus hook (Phase 2 Unit 8)
#
# Invoked by tmux on pane-focus-in/pane-focus-out/window-focused events.
# Writes a single JSONL line to ~/.ambient/focus-events.jsonl.
#
# Privacy contract (cites docs/PRIVACY.md clauses 6, 7):
# - Captured fields: hook event name, pane id, window index, session name, ts.
# - NEVER captured: pane_current_command, pane_current_path, pane_title,
#   pane_current_pid. tmux exposes these via format strings; this hook
#   intentionally references none of them.
# - Off by default; installed and removed by `ambient tmux-focus-enable` /
#   `ambient tmux-focus-disable`.
#
# Marker comment for idempotent install/remove: # ambient-managed

set -eu

EVENT_NAME="${1:-unknown}"
EVENTS_PATH="${AMBIENT_FOCUS_EVENTS_PATH:-$HOME/.ambient/focus-events.jsonl}"

# Resolve fields from the tmux environment. tmux exports $TMUX_PANE
# unconditionally inside `run-shell`. session name + window index come via
# `tmux display-message -p`, which we call only with structural format
# strings (#S session name, #I window index).
PANE_ID="${TMUX_PANE:-}"
SESSION_NAME=""
WINDOW_INDEX=""
if command -v tmux >/dev/null 2>&1 && [ -n "${TMUX:-}" ]; then
    SESSION_NAME="$(tmux display-message -p '#S' 2>/dev/null || echo "")"
    WINDOW_INDEX="$(tmux display-message -p '#I' 2>/dev/null || echo "")"
fi

# ISO 8601 UTC timestamp, second precision.
TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

mkdir -p "$(dirname "$EVENTS_PATH")"

# JSON-escape any field that could legitimately contain quotes or backslashes.
# tmux session names allow most printable chars including quotes, brackets,
# and (rarely) backslashes; emoji are common. Without escaping, a session
# named `it's mine` produces invalid JSON that the reader silently drops.
# This is a minimal escaper for the four chars that break a JSON string body
# in the contexts we care about: backslash, double-quote, newline, tab.
json_escape() {
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' \
                            -e 's/	/\\t/g' \
                            -e ':a;N;$!ba;s/\n/\\n/g'
}

ESC_EVENT=$(json_escape "$EVENT_NAME")
ESC_PANE=$(json_escape "$PANE_ID")
ESC_WINDOW=$(json_escape "$WINDOW_INDEX")
ESC_SESSION=$(json_escape "$SESSION_NAME")

# Build JSONL line. Field set is fixed.
printf '{"ts":"%s","source":"tmux","event":"%s","pane_id":"%s","window_index":"%s","session_name":"%s"}\n' \
    "$TS" \
    "$ESC_EVENT" \
    "$ESC_PANE" \
    "$ESC_WINDOW" \
    "$ESC_SESSION" \
    >> "$EVENTS_PATH"
