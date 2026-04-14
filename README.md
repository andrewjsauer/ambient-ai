# Ambient AI

Passive terminal behavioral monitor and development coaching system for macOS. Captures shell commands and Claude Code conversations, runs algorithmic pattern detection, and produces daily/weekly coaching reports with actionable recommendations.

## What It Does

Ambient AI watches two things:

1. **Your terminal** -- every command, exit code, and timing gap via zsh hooks
2. **Your Claude Code sessions** -- prompts you type, tools Claude uses, errors, files touched

It runs 7 detectors over this data and produces coaching output: daily summaries, weekly trend reports, on-demand insights, and installable recommendations (skills, aliases, CLAUDE.md rules).

The key differentiator is **resolution velocity tracking** -- Ambient AI is the only tool that sees the full debugging loop (shell failure -> Claude session -> fix attempt -> shell retry -> success) and measures how fast you resolve problems, where you get stuck, and what you should change.

## Architecture

```
CAPTURE ──> DETECT ──> PRESENT
```

**Capture** (zsh hooks + session parser) writes events to `~/.ambient/logs/events-YYYY-MM-DD.jsonl`.

**Detect** (7 algorithmic detectors, no LLM) produces structured findings:

| Detector | What it finds |
|----------|--------------|
| Compression | Repeated command sequences (alias candidates) |
| Pauses | Cognitive states: routine, evaluating, stuck (GMM classifier) |
| Changepoints | Workflow rhythm shifts (PELT algorithm) |
| Projects | Per-project time allocation and context switches |
| Prompt Patterns | Repeated Claude prompts (skill candidates) |
| Coaching | Session outcomes: Productive, Friction, Quick, Abandoned + thrash scores |
| Velocity | Resolution chains: fail -> Claude -> success, measured in active time |

**Present** (Haiku/Sonnet synthesis) generates narratives:

| Output | Cadence | Model |
|--------|---------|-------|
| Batch analysis | Every 30 min (daemon) | Haiku |
| Daily summary | End of day (daemon) | Sonnet |
| Weekly digest | Sunday (daemon) | Sonnet |
| Coaching report | On-demand (`ambient insights`) | Sonnet |
| Recommendations | Daily (daemon) | Haiku |
| Stuck notifications | Real-time (daemon) | None (macOS native) |

## Quick Start

### Prerequisites

- macOS (uses launchd for scheduling, osascript for notifications)
- Python 3.10+
- zsh (default macOS shell)
- Anthropic API key

### Install

```bash
git clone https://github.com/andrewjsauer/ambient-ai.git
cd ambient-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

```bash
# Create .env with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
chmod 600 .env
```

### Set Up Shell Hooks

Add to your `~/.zprofile` or `~/.zshrc`:

```bash
source /path/to/ambient-ai/src/ambient/capture/hooks.zsh
alias ambient='/path/to/ambient-ai/.venv/bin/ambient'
```

Start a new shell or `source ~/.zprofile`.

### Start the Daemon

```bash
source .env && export ANTHROPIC_API_KEY && ambient daemon-start
```

The daemon ticks every 30 minutes, ingesting sessions, running detectors, and generating summaries automatically.

### Verify

```bash
ambient daemon-status   # should show "Daemon: running"
ambient status          # shows today's event count
```

## CLI Commands

### Analysis

| Command | Description |
|---------|-------------|
| `ambient stats [--window MIN]` | Raw detector output (no LLM) |
| `ambient analyze` | Run batch analysis with Haiku narration |
| `ambient summary [--date DATE]` | Generate daily summary |
| `ambient review [--date DATE]` | View saved daily summary |
| `ambient insights [--window DAYS]` | Coaching report with velocity + stuck patterns |
| `ambient projects [--window MIN \| --date DATE]` | Per-project time allocation |

### Recommendations

| Command | Description |
|---------|-------------|
| `ambient recommendations` | List pending recommendations |
| `ambient apply <id>` | Install a skill to `~/.claude/commands/` |

### Daemon

| Command | Description |
|---------|-------------|
| `ambient daemon-start` | Register launchd agent |
| `ambient daemon-stop` | Unload launchd agent |
| `ambient daemon-status` | Running status, cursors, lock |

### Setup

| Command | Description |
|---------|-------------|
| `ambient start` | Show setup instructions |
| `ambient stop` | Show teardown instructions |
| `ambient status` | Event count, calibration status |
| `ambient calibrate` | Fit GMM on accumulated data |

## Data Flow

### Daemon Tick Cycle (every 30 minutes)

1. Load API key from `~/.ambient/.env`
2. Acquire PID-based lock (stale detection at 60 min)
3. Ingest completed Claude Code sessions (incremental -- tracks line counts for long-lived sessions)
4. Read new events since last cursor
5. Run detectors + Haiku batch analysis -> write to `analysis-YYYY-MM-DD.jsonl`
6. Advance cursor atomically
7. Check for missing daily summaries -> generate with Sonnet
8. Check coaching recommendations -> stage with quality gate
9. Check weekly summary (Sundays) -> generate with coaching section
10. Auto-recalibrate GMM if eligible (7 days + 200 events)
11. Release lock

### Session Ingestion

Claude Code writes per-session JSONL files to `~/.claude/projects/<slug>/<uuid>.jsonl`. The daemon:

- Discovers all session files across project directories
- Waits 30 minutes after last file modification (session considered complete)
- Parses incrementally: tracks line count per session, only extracts new prompts/tools/errors on subsequent passes
- Supports long-lived sessions that span hours or days

### Coaching System

The coaching system (Tier 2.5) adds three signals on top of the base detectors:

**Session Outcome Classification** -- Each Claude session is labeled using heuristic precedence:
1. Quick (< 5 prompts, < 3 tool calls)
2. Abandoned (> 1 prompt, no Write/Edit, errors present, > 5 min)
3. Friction (thrash score > 0.5)
4. Productive (everything else)

**Thrash Score** -- `error_count / prompt_count` per session (floor: 3 prompts). High scores indicate Claude-mediated stuck loops.

**Resolution Velocity** -- Detects fail -> Claude -> success chains:
- Measures active time (not wall clock)
- 15-minute idle gaps break the chain
- Subsequent command must match the failed command type and project
- Per-project breakdown with avg, median, p90

### Recommendation Engine

Two recommendation paths, both staged to `~/.ambient/recommendations/`:

**Prompt-pattern recommendations** -- When a normalized prompt appears 5+ times across sessions, Haiku drafts a Claude Code skill definition.

**Coaching recommendations** -- Quality-gated (3+ stuck episodes on same project, OR velocity > 2x average, OR pattern across 3+ sessions). Haiku drafts CLAUDE.md rules.

Install with `ambient apply <id>` which copies skills to `~/.claude/commands/`.

## File Layout

### Runtime Data (`~/.ambient/`)

```
~/.ambient/
  .env                              # ANTHROPIC_API_KEY (chmod 600)
  logs/
    events-YYYY-MM-DD.jsonl         # raw captured events
  analysis/
    analysis-YYYY-MM-DD.jsonl       # batch detector findings
    summary-YYYY-MM-DD.md           # daily narrative
    weekly-YYYY-MM-DD.md            # weekly trend report
  insights/
    insights-YYYY-MM-DD.md          # coaching report
  recommendations/
    skill-<name>.md                 # generated skill definitions
    coaching-<name>.md              # generated CLAUDE.md rules
  models/
    gmm.joblib                      # fitted pause classifier
  daemon/
    state.json                      # cursor, processed sessions
    daemon.lock                     # PID lock
    daemon.log                      # rotating log (7 days)
```

### Source Code

```
src/ambient/
  cli.py                            # 15 CLI commands
  config.py                         # all tunable parameters
  capture/
    hooks.zsh                       # zsh preexec/precmd hooks
    reader.py                       # Event dataclass + JSONL reader
  detect/
    compression.py                  # repeated command sequences
    pauses.py                       # GMM pause classifier
    changepoints.py                 # PELT rhythm detection
    projects.py                     # per-project time allocation
    prompt_patterns.py              # normalized prompt grouping
    correlator.py                   # shell <-> Claude event linking
    coaching.py                     # session outcomes + thrash scoring
    velocity.py                     # resolution chain detection
  present/
    api.py                          # Anthropic API wrapper
    narrator.py                     # batch/daily/weekly synthesis
    prompts.py                      # system prompts + data formatting
    insights.py                     # coaching report generation
    recommender.py                  # skill/alias/CLAUDE.md generation
    notify.py                       # macOS stuck notifications
    tokens.py                       # token estimation for budgeting
  daemon/
    tick.py                         # 30-min daemon cycle
    state.py                        # persistent cursor state
    lock.py                         # PID-based concurrency control
    launchd.py                      # macOS agent registration
    session_parser.py               # Claude session JSONL parser
```

## Configuration

All parameters are in `src/ambient/config.py` as dataclass fields with sensible defaults:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `session_boundary_ms` | 600,000 (10 min) | Gap threshold for new terminal session |
| `min_sequence_frequency` | 3 | Minimum repetitions for compression detection |
| `gmm_n_components` | 3 | Pause classifier mixture components |
| `thrash_score_threshold` | 0.5 | Friction classification threshold |
| `thrash_min_prompts` | 3 | Minimum prompts before computing thrash score |
| `velocity_idle_break_ms` | 900,000 (15 min) | Idle gap that breaks a resolution chain |
| `velocity_min_chains` | 5 | Minimum chains for meaningful velocity metrics |
| `weekly_min_weeks` | 2 | Weeks of data required before weekly digest |
| `haiku_model` | claude-haiku-4-5 | Model for batch analysis + skill generation |
| `sonnet_model` | claude-sonnet-4-6 | Model for daily/weekly/insights synthesis |

## API Costs

Approximate daily costs with normal usage:

| Call | Model | Frequency | Est. Cost |
|------|-------|-----------|-----------|
| Batch analysis | Haiku | ~16/day (every 30 min during active hours) | ~$0.02 |
| Daily summary | Sonnet | 1/day | ~$0.03 |
| Skill generation | Haiku | 0-3/day | ~$0.01 |
| Coaching recs | Haiku | 0-2/day | ~$0.01 |
| `ambient insights` | Sonnet | On-demand | ~$0.05 |
| Weekly digest | Sonnet | 1/week | ~$0.04 |

**Total: ~$0.05-0.15/day** with active terminal use. Zero cost when idle (no API calls).

## Sleep/Wake Behavior

- **Laptop closed:** launchd suspends all jobs. Nothing runs. No queued backlog.
- **Wake up:** Next scheduled tick fires. One tick processes all accumulated events since sleep.
- **Idle terminal:** Daemon ticks, finds no new events, exits in <1 second. No API calls.
- **Long-lived Claude sessions:** Incremental ingestion picks up new content when the session goes idle for 30 minutes, even if the session has been open for hours.

## Dependencies

- `anthropic` -- Claude API client
- `scikit-learn` -- Gaussian Mixture Model for pause classification
- `ruptures` -- PELT changepoint detection
- `numpy` -- numerical computing
- `joblib` -- model persistence
- `python-dotenv` -- API key loading

## License

Private repository.
