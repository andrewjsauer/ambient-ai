# Auto-Scheduling Daemon

The daemon automates ambient-ai's analysis pipeline so you never need to manually run `ambient analyze`, `ambient summary`, or `ambient calibrate`.

## Quick Start

```bash
# 1. Make sure your API key is set in the current shell
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Start the daemon (registers with macOS launchd)
ambient daemon-start

# 3. Check it's running
ambient daemon-status

# 4. That's it. Analysis runs every 30 minutes automatically.
#    Check results anytime:
ambient review
```

## How It Works

The daemon is a macOS launchd user agent that invokes `ambient daemon-tick` every 30 minutes. Each tick follows a gate sequence that exits early when there's nothing to do.

### Gate Sequence (cheapest check first)

1. **API key** -- `load_dotenv(~/.ambient/.env)`, check `ANTHROPIC_API_KEY`. If missing, log and exit.
2. **New events** -- Read events since the last cursor position. If none, skip analysis but still check for missing summaries.
3. **Lock** -- Acquire PID-based lock file. If another tick is running, exit.

### What Each Tick Does

When gates pass:

1. **Analyze** -- Runs the full detection pipeline (compression + pause classification) on new events, sends findings to Claude Haiku, appends results to `analysis-YYYY-MM-DD.jsonl`
2. **Update cursor** -- Advances the watermark to `latest_event.ts_start + 1` (exclusive bound) so events are never double-processed
3. **Save state** -- Persists cursor immediately after analysis (crash-safe)
4. **Catch-up summaries** -- Scans from `last_summary_date` forward for any day with analysis data but no summary file. Generates missing summaries using the full pipeline (changepoint detection + Claude Sonnet). Handles weekends, vacations, and gaps of any length.
5. **Recalibration check** -- If >=7 days since last calibration AND >=200 new events, re-fits the GMM pause classifier on all accumulated data
6. **Release lock**

### Cursor-Based Watermark

Unlike the manual `ambient analyze` (which uses a wall-clock 30-minute window), the daemon tracks a cursor -- the timestamp of the last processed event. This eliminates gaps and overlaps from launchd timing jitter. If the machine sleeps and wakes, the daemon processes all accumulated events on the next tick.

### API Key Handling

launchd agents don't inherit shell environment variables. The daemon solves this by:

- `ambient daemon-start` copies the current `ANTHROPIC_API_KEY` to `~/.ambient/.env` (chmod 600)
- Each tick calls `load_dotenv()` to read the key
- Running `daemon-start` again overwrites the key (handles rotation)

### Lock File

`~/.ambient/daemon/daemon.lock` prevents concurrent ticks. Uses PID + stale detection:

- If the lock PID is alive, the tick backs off
- If the lock PID is dead (crash), the lock is broken and re-acquired
- All operations (analysis, summaries, recalibration) run inside the lock

### State File

`~/.ambient/daemon/state.json` tracks:

```json
{
  "last_analyzed_ts": 1711870984117,
  "last_summary_date": "2026-03-30",
  "last_calibration_date": "2026-03-24",
  "events_since_calibration": 450
}
```

Saved atomically (write to temp file, then `os.replace`). Handles corrupt/missing state gracefully (defaults to empty state on the next tick).

## Structured Daily Summaries

Daily summaries use a fixed template with 8 sections, inspired by Claude Code's session memory system. This makes summaries consistent across days and machine-parseable for future longitudinal analysis.

```markdown
## Day Title
_A distinctive 5-10 word summary of this workday._

## Rhythm Profile
_When were you most focused vs most fragmented?_

## Automation Candidates
_Top 3 repeated command sequences, ranked by time saved._

## Cognitive Load
_Stuck episodes, routine vs evaluating vs stuck ratio._

## Workflow Phases
_Chronological phases with dominant activity type._

## Friction Points
_Failed commands, retries, longest stuck episodes._

## Key Stats
_Event count, session count, command rate, stuck count._

## Actionable Insight
_One specific suggestion for tomorrow._
```

Each section's italic description is preserved in the output as a structural anchor. Sections are left blank when insufficient data exists.

## CLI Commands

| Command | What it does |
|---------|-------------|
| `ambient daemon-start` | Registers launchd agent, saves API key to `~/.ambient/.env` |
| `ambient daemon-stop` | Unloads launchd agent |
| `ambient daemon-status` | Shows: running/stopped, last analysis time, last summary, calibration state, lock status, today's event count |
| `ambient daemon-tick` | Hidden. The launchd entry point. Runs one tick cycle. |

## File Layout

```
~/.ambient/
  .env                          # API key (created by daemon-start, chmod 600)
  logs/
    events-YYYY-MM-DD.jsonl     # Raw shell events (written by zsh hooks)
  analysis/
    analysis-YYYY-MM-DD.jsonl   # Batch analysis results (appended by daemon)
    summary-YYYY-MM-DD.md       # Daily summaries (written by daemon)
  models/
    gmm.joblib                  # Pause classifier model
  daemon/
    state.json                  # Daemon state (cursor, dates, counters)
    daemon.lock                 # PID lock file
    daemon.log                  # Daemon activity log (7-day rotation)
    launchd-stdout.log          # launchd stdout capture
    launchd-stderr.log          # launchd stderr capture

~/Library/LaunchAgents/
  com.ambient.daemon.plist      # launchd agent definition
```

## Source Code Layout

```
src/ambient/
  cli.py              # CLI entry point, 12 commands including daemon-*
  config.py            # Config dataclass with all paths
  daemon/
    __init__.py
    tick.py            # daemon_tick() -- gate sequence, analysis, summaries, recal
    state.py           # DaemonState dataclass with atomic JSON persistence
    lock.py            # PID-based lock file (acquire, release, is_locked)
    launchd.py         # Plist generation, launchctl bootstrap/bootout
  capture/
    hooks.zsh          # zsh preexec/precmd hooks
    reader.py          # Event dataclass, JSONL reader, time-range queries
  detect/
    compression.py     # Repeated sequence detection
    pauses.py          # GMM pause classifier + calibration
    changepoints.py    # PELT changepoint detection
  present/
    narrator.py        # Claude API calls, JSONL/MD output
    prompts.py         # System + batch + daily prompt templates
```

## Troubleshooting

**Daemon not processing events:**
```bash
ambient daemon-status     # Check if running and when last analysis happened
cat ~/.ambient/daemon/daemon.log | tail -20   # Check for errors
```

**API key rotated:**
```bash
export ANTHROPIC_API_KEY=sk-ant-new-key
ambient daemon-start      # Overwrites ~/.ambient/.env with new key
```

**Daemon stuck (lock file stale):**
The lock auto-breaks when the holding PID is dead. If the PID is alive but hung, wait 60 minutes or manually remove `~/.ambient/daemon/daemon.lock`.

**Missing summaries for past days:**
The daemon catch-up scan generates summaries for any day with analysis data but no summary file. It runs on every tick, so missing summaries are filled in automatically.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| launchd over Python daemon | launchd handles scheduling, restart, and login-session binding natively. A Python daemon adds signal handling and watchdog complexity for no benefit. |
| Cursor over fixed window | Fixed 30-min windows gap/overlap with launchd timing jitter. Cursor-based approach processes exactly the events that haven't been seen yet. |
| `load_dotenv` over launchd env vars | launchd agents don't inherit shell env. `launchctl setenv` doesn't persist across reboots. `load_dotenv` from `~/.ambient/.env` is reliable and uses an existing dependency. |
| `launchctl bootstrap/bootout` over `load/unload` | `load/unload` deprecated on macOS 13+. Modern API avoids deprecation warnings. |
| Flat CLI commands over nested subparsers | Matches existing argparse pattern. `daemon-start` is one token, no nesting complexity. |
| Lock protects all operations | Analysis, summaries, and recalibration all run inside the lock. Prevents any concurrent mutation. |
| State saved immediately after cursor | If the daemon crashes during summaries/recalibration, the cursor is already persisted. Events won't be re-analyzed. |
| try/except around summaries/recalibration | Transient errors (API failure, corrupt file) don't crash the daemon. The tick completes and tries again next time. |
