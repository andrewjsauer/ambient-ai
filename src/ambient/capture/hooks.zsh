# Ambient AI - Shell hooks for event capture
# Source this file in your .zshrc: source /path/to/hooks.zsh
# Requires: zsh 5.8+ (for $EPOCHREALTIME), tmux (optional)

# Guard against recursive sourcing
[[ -n "$_AMBIENT_LOADED" ]] && return
_AMBIENT_LOADED=1

# Load required modules
zmodload zsh/datetime    # provides $EPOCHREALTIME
autoload -Uz add-zsh-hook

# Configuration
_AMBIENT_LOG_DIR="${AMBIENT_LOG_DIR:-$HOME/.ambient/logs}"
_AMBIENT_IGNORE_CMDS="${AMBIENT_IGNORE_CMDS:-ls cd clear pwd echo}"
_AMBIENT_SESSION_BOUNDARY_MS="${AMBIENT_SESSION_BOUNDARY_MS:-600000}"

# State variables
_AMBIENT_CMD_START=
_AMBIENT_CMD=
_AMBIENT_CWD=
_AMBIENT_PANE=
_AMBIENT_LAST_END=

# Get current time in epoch milliseconds using EPOCHREALTIME (zsh 5.8+)
# No subshells, no date command - pure zsh builtins
_ambient_now_ms() {
    local rt="$EPOCHREALTIME"
    local secs="${rt%.*}"
    local frac="${rt#*.}"
    # Take first 3 digits of fractional part for milliseconds
    frac="${frac:0:3}"
    # Pad with zeros if fractional part is short
    while [[ ${#frac} -lt 3 ]]; do
        frac="${frac}0"
    done
    REPLY=$(( secs * 1000 + frac ))
}

# Check if command should be ignored
_ambient_should_ignore() {
    local cmd_name="${1%% *}"
    local ignore
    for ignore in ${=_AMBIENT_IGNORE_CMDS}; do
        [[ "$cmd_name" == "$ignore" ]] && return 0
    done
    return 1
}

# Get tmux pane ID if running in tmux
_ambient_pane_id() {
    if [[ -n "$TMUX" ]]; then
        REPLY=$(tmux display-message -p '#{pane_id}' 2>/dev/null) || REPLY=""
    else
        REPLY=""
    fi
}

# Escape a string for JSON (handles quotes, backslashes, newlines)
_ambient_json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\t'/\\t}"
    s="${s//$'\r'/\\r}"
    REPLY="$s"
}

# preexec: fires before each command
_ambient_preexec() {
    local cmd="$1"

    # Check ignore list
    _ambient_should_ignore "$cmd" && { _AMBIENT_CMD=; return; }

    # Record start time and command
    _ambient_now_ms
    _AMBIENT_CMD_START=$REPLY
    _AMBIENT_CMD="$cmd"
    _AMBIENT_CWD="$PWD"
    _ambient_pane_id
    _AMBIENT_PANE="$REPLY"
}

# precmd: fires after command completion, before next prompt
_ambient_precmd() {
    local exit_code=$?

    # Skip if no command was captured (filtered or empty)
    [[ -z "$_AMBIENT_CMD" ]] && return

    # Compute end time
    _ambient_now_ms
    local ts_end=$REPLY
    local duration_ms=$(( ts_end - _AMBIENT_CMD_START ))

    # Compute gap from last command
    local gap_ms=""
    local session_boundary="false"
    if [[ -n "$_AMBIENT_LAST_END" ]]; then
        gap_ms=$(( _AMBIENT_CMD_START - _AMBIENT_LAST_END ))
        if (( gap_ms > _AMBIENT_SESSION_BOUNDARY_MS )); then
            session_boundary="true"
        fi
    fi

    # Update last end time
    _AMBIENT_LAST_END=$ts_end

    # Ensure log directory exists
    [[ -d "$_AMBIENT_LOG_DIR" ]] || mkdir -p "$_AMBIENT_LOG_DIR"

    # Build log file path
    local date_str
    strftime -s date_str "%Y-%m-%d" "${EPOCHREALTIME%.*}"
    local log_file="${_AMBIENT_LOG_DIR}/events-${date_str}.jsonl"

    # Escape command and cwd for JSON
    _ambient_json_escape "$_AMBIENT_CMD"
    local escaped_cmd="$REPLY"
    _ambient_json_escape "$_AMBIENT_CWD"
    local escaped_cwd="$REPLY"

    # Build JSON line
    local json_line="{\"ts_start\":${_AMBIENT_CMD_START},\"ts_end\":${ts_end},\"duration_ms\":${duration_ms},\"command\":\"${escaped_cmd}\",\"exit_code\":${exit_code},\"cwd\":\"${escaped_cwd}\""

    # Add tmux pane if available
    if [[ -n "$_AMBIENT_PANE" ]]; then
        json_line="${json_line},\"tmux_pane\":\"${_AMBIENT_PANE}\""
    else
        json_line="${json_line},\"tmux_pane\":null"
    fi

    # Add gap_ms
    if [[ -n "$gap_ms" ]]; then
        json_line="${json_line},\"gap_ms\":${gap_ms}"
        if [[ "$session_boundary" == "true" ]]; then
            json_line="${json_line},\"session_boundary\":true"
        fi
    else
        json_line="${json_line},\"gap_ms\":null"
    fi

    json_line="${json_line}}"

    # Append to log file (>> is atomic for small writes on APFS)
    print -r -- "$json_line" >> "$log_file"

    # Clear command state
    _AMBIENT_CMD=
}

# Register hooks
add-zsh-hook preexec _ambient_preexec
add-zsh-hook precmd _ambient_precmd
