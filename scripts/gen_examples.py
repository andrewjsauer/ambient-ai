#!/usr/bin/env python3
"""Generate docs/EXAMPLES.md — output of every analysis command, from SYNTHETIC
data only. Runs the real `cmd_*` functions against a throwaway base dir; never
reads ~/.ambient or ~/.claude, makes no API calls, and commits no real data.

    python scripts/gen_examples.py
"""

import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ambient.config import Config
from ambient.present.insights import aggregate_coaching_data, format_terminal_summary

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import make_projects, synthetic_events  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "EXAMPLES.md"
MIN_MS = 60_000


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue().rstrip("\n")


def _seed_runtime(config, now_ms):
    """Make the dashboard look like a live, calibrated, daemon-running setup."""
    from ambient.daemon.state import DaemonState

    today = datetime.now().strftime("%Y-%m-%d")
    config.summary_path(today).write_text("# Daily Summary\n\nProductive day.\n")
    state = DaemonState(
        last_analyzed_ts=now_ms - 4 * MIN_MS,
        last_summary_date=today,
        events_since_calibration=180,
    )
    state.save(config.state_path)

    recs = config.recommendations_dir
    recs.mkdir(parents=True, exist_ok=True)
    (recs / "skill-add-regression-test.md").write_text(
        '---\ntype: skill\ntitle: "Skill: add a regression test for the failing case"\n---\nbody\n')
    (recs / "alias-git-wip.md").write_text(
        '---\ntype: alias\ntitle: "Alias: gwip = git add -A && git commit -m wip && git push"\n---\nbody\n')


def main():
    import ambient.cli as cli

    now = datetime.now()
    now_ms = int(now.timestamp() * 1000)

    # Projects live at a clean, generic path (not the ugly $TMPDIR) because the
    # stats view prints session cwds verbatim. Created real on disk so
    # project_capabilities can detect test targets; removed in the finally.
    clean_projects = Path("/tmp/ambient-demo-projects")
    shutil.rmtree(clean_projects, ignore_errors=True)
    clean_projects.mkdir(parents=True)

    sections = []
    try:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            empty_projects = base / "claude_projects"
            empty_projects.mkdir()
            proj = make_projects(clean_projects)
            events = synthetic_events(now_ms, proj)
            # gmm_min_samples lowered for the demo only (the synthetic slice has
            # ~29 gaps vs the real 60 floor) so the pause classifier calibrates
            # and the stats example shows real labels.
            config = Config(base_dir=base, claude_projects_dir=empty_projects,
                            gmm_min_samples=20)
            config.ensure_dirs()

            by_day = {}
            for e in events:
                ds = datetime.fromtimestamp(e["ts_end"] / 1000).strftime("%Y-%m-%d")
                by_day.setdefault(ds, []).append(e)
            for ds, day_events in by_day.items():
                with open(config.events_path(ds), "w") as f:
                    for e in day_events:
                        f.write(json.dumps(e) + "\n")

            # Fit a real GMM so the pause classifier is calibrated in the examples.
            from ambient.capture.reader import read_events
            from ambient.detect.pauses import calibrate
            all_ev = read_events(config, start=datetime.now() - timedelta(days=2),
                                 end=datetime.now() + timedelta(minutes=1))
            calibrate(all_ev, config)

            _seed_runtime(config, now_ms)

            # status (patch the one system probe so the demo is deterministic)
            with patch("ambient.daemon.launchd.is_agent_loaded", return_value=True):
                sections.append(("ambient status",
                                 "Daemon health, today's activity, and what to run next.",
                                 _capture(cli.cmd_status, config, SimpleNamespace())))

            # insights (local terminal summary — the coaching report headline)
            data = aggregate_coaching_data(config, window_days=7)
            sections.append(("ambient insights",
                             "The coaching report: resolution velocity, stuck patterns, "
                             "repeated prompts, and verification gaps.",
                             format_terminal_summary(data)))

            sections.append(("ambient projects --window 2880",
                             "Per-project time allocation and context switches.",
                             _capture(cli.cmd_projects, config, SimpleNamespace(window=2880, date=None))))

            sections.append(("ambient stats --window 2880",
                             "Raw algorithmic detector output — no LLM involved.",
                             _capture(cli.cmd_stats, config, SimpleNamespace(window=2880))))

            sections.append(("ambient recommendations",
                             "Installable skill / alias drafts staged from your patterns.",
                             _capture(cli.cmd_recommendations, config, SimpleNamespace())))
    finally:
        shutil.rmtree(clean_projects, ignore_errors=True)

    body = [
        "# Example output",
        "",
        "Every block below is produced by `scripts/gen_examples.py` from **synthetic** "
        "data — invented project names, prompts, and paths. No real `~/.ambient` or "
        "`~/.claude` data is read or committed. Regenerate with:",
        "",
        "```bash",
        "python scripts/gen_examples.py",
        "```",
        "",
    ]
    for cmd, blurb, out in sections:
        body += [f"## `{cmd}`", "", blurb, "", "```", out, "```", ""]
    OUT.write_text("\n".join(body))
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes, {len(sections)} commands)")


if __name__ == "__main__":
    main()
