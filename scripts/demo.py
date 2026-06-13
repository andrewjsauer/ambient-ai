#!/usr/bin/env python3
"""Generate the README demo output from SYNTHETIC data.

Builds a throwaway event log under a temp directory, runs the real detector
pipeline against it (no API key required, no network), and prints the same
`ambient insights` terminal summary a user would see. Every project name,
prompt, and path here is invented — this script never reads ~/.ambient or
~/.claude, so it is safe to run and safe to commit its output.

    python scripts/demo.py            # print the summary
    python scripts/demo.py --json     # also dump the synthetic events

Regenerate docs/assets/demo.svg with scripts/render_demo_svg.py.
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from ambient.config import Config
from ambient.present.insights import aggregate_coaching_data, format_terminal_summary

DAY_MS = 86_400_000
MIN_MS = 60_000


def _cmd(ts, command, cwd, exit_code=0, duration_ms=4_000, gap_ms=2_000):
    return {
        "ts_start": ts, "ts_end": ts + duration_ms, "duration_ms": duration_ms,
        "command": command, "exit_code": exit_code, "cwd": cwd,
        "tmux_pane": "%0", "gap_ms": gap_ms, "session_boundary": False,
        "type": "command",
    }


def _session(ts, project, prompts, tools, files, duration_ms=8 * MIN_MS,
             error_count=0):
    return {
        "type": "claude_session", "ts_start": ts, "ts_end": ts + duration_ms,
        "duration_ms": duration_ms,
        "command": f"claude: {prompts[0]}", "exit_code": 0,
        "cwd": project, "tmux_pane": None, "gap_ms": None,
        "claude_session_id": f"sess-{ts}",
        "claude_prompts": prompts,
        "claude_tools": [{"name": t, "files": files} for t in tools],
        "claude_files": files,
        "claude_project": project,  # full path, exactly as real ingestion writes it
        "claude_prompt_count": len(prompts),
        "claude_is_error_count": error_count,
    }


def make_projects(root: Path) -> dict:
    """Create real project dirs with manifests so project_capabilities can
    detect test/typecheck targets — that turns the verification-gap section
    from 'non-verifiable' into real coverage rates. Returns name -> abs path."""
    paths = {}
    for name in ("payments-api", "web-app", "data-pipeline", "infra"):
        d = root / name
        d.mkdir()
        paths[name] = str(d)
    # Python projects with pytest configured.
    for name in ("payments-api", "data-pipeline"):
        (root / name / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n")
    # Node project with test + typecheck scripts.
    (root / "web-app" / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "typecheck": "tsc --noEmit", "dev": "vite"}}))
    (root / "web-app" / "tsconfig.json").write_text("{}")
    # infra: terraform, genuinely no test/typecheck target — stays "non-verifiable".
    return paths


def _chain(ev, t, project, test_cmd, prompts, active_min):
    """fail -> multi-turn Claude session (with an Edit) -> passing test soon
    after, so the fix is both a resolved velocity chain and a *verified* fix.
    `active_min` is the Claude session length (the bulk of active debugging
    time); the verifying test fires 90s after the session ends, inside the
    5-min verification window."""
    sess_dur = active_min * MIN_MS
    ev.append(_cmd(t, test_cmd, project, exit_code=1))
    ev.append(_session(t + 30_000, project, prompts, ["Read", "Edit", "Bash"],
                       [f"{test_cmd.split()[0]}_target.py"], duration_ms=sess_dur))
    ev.append(_cmd(t + 30_000 + sess_dur + 90_000, test_cmd, project, exit_code=0))


def synthetic_events(now_ms, proj):
    """A believable two-day slice: resolved debugging loops, one stuck project,
    repeated prompts, and a repeated git sequence."""
    ev = []
    # Seven fail -> Claude -> verified-success chains across three projects.
    chains = [
        (proj["payments-api"], "pytest", ["fix the failing charge-refund test",
            "still red on the partial-refund case", "good, add a regression test"], 14),
        (proj["payments-api"], "pytest", ["why does the webhook retry loop",
            "it double-charges on 5xx", "guard with an idempotency key"], 9),
        (proj["payments-api"], "pytest", ["rounding is off on invoice totals",
            "use banker's rounding", "confirm the cents add up"], 6),
        (proj["web-app"], "npm test", ["the checkout button fires twice",
            "debounce or disable on submit", "disable then re-enable on settle"], 22),
        (proj["web-app"], "npm test", ["cart total wrong after coupon removal",
            "recompute from line items", "memo the selector"], 11),
        (proj["data-pipeline"], "pytest", ["dedupe the nightly ingest rows",
            "dupes come from the retry path", "key on event id not timestamp"], 18),
        (proj["data-pipeline"], "pytest", ["null dates crash the loader",
            "coerce to epoch or skip", "skip and log the row"], 7),
    ]
    t = now_ms - DAY_MS
    for project, test_cmd, prompts, active_min in chains:
        _chain(ev, t, project, test_cmd, prompts, active_min)
        t += 90 * MIN_MS

    # One project that keeps getting stuck: three high-thrash, no-fix sessions
    # (no Edit -> abandoned, and terraform has no test target).
    t = now_ms - 6 * 60 * MIN_MS
    for _ in range(3):
        ev.append(_cmd(t, "terraform apply", proj["infra"], exit_code=1))
        ev.append(_session(
            t + 20_000, proj["infra"],
            ["the state lock won't release", "still locked, what now",
             "is force-unlock safe here", "ok try force-unlock"],
            ["Read", "Grep", "Bash"], ["main.tf"],
            duration_ms=16 * MIN_MS, error_count=5,
        ))
        t += 40 * MIN_MS

    # Three linter-loop fixes with no follow-up test -> real verification gaps
    # AND a repeated prompt (skill candidate).
    t = now_ms - 5 * 60 * MIN_MS
    for _ in range(4):
        ev.append(_session(t, proj["web-app"],
                          ["run the linter and fix everything"], ["Bash", "Edit"], ["app.tsx"],
                          duration_ms=3 * MIN_MS))
        t += 25 * MIN_MS

    # A repeated command sequence (alias candidate).
    t = now_ms - 3 * 60 * MIN_MS
    for _ in range(4):
        for c in ("git add -A", "git commit -m wip", "git push"):
            ev.append(_cmd(t, c, proj["web-app"]))
            t += 30_000
        t += 10 * MIN_MS

    ev.sort(key=lambda e: e["ts_start"])
    return ev


def main():
    now = datetime.now()
    now_ms = int(now.timestamp() * 1000)

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        empty_projects = base / "claude_projects"  # keep phase-1 detectors off real data
        empty_projects.mkdir()
        projects_root = base / "projects"
        projects_root.mkdir()
        proj = make_projects(projects_root)
        events = synthetic_events(now_ms, proj)
        config = Config(base_dir=base, claude_projects_dir=empty_projects)
        config.ensure_dirs()

        # Write events into the day files the reader expects.
        by_day = {}
        for e in events:
            ds = datetime.fromtimestamp(e["ts_end"] / 1000).strftime("%Y-%m-%d")
            by_day.setdefault(ds, []).append(e)
        for ds, day_events in by_day.items():
            with open(config.events_path(ds), "w") as f:
                for e in day_events:
                    f.write(json.dumps(e) + "\n")

        data = aggregate_coaching_data(config, window_days=7)
        print(format_terminal_summary(data))

    if "--json" in sys.argv:
        print("\n--- synthetic events ---", file=sys.stderr)
        print(json.dumps(events, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
