# Contributing to ambient-ai

## Setup

```bash
git clone https://github.com/andrewjsauer/ambient-ai.git
cd ambient-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

An editable install from a clone is the supported mode — the macOS
notification app and the tmux hook script resolve paths relative to the
repository.

## Running Tests

```bash
pytest
```

The full suite runs in under 30 seconds with no network access and no API
key. Every behavioral change should land with tests; bug fixes should include
a regression test that fails on the pre-fix code.

## Conventions

- **Commits** follow conventional-commit style: `type(scope): summary` —
  lowercase, imperative (e.g. `fix(velocity): derive project basename for
  claude_session events`). Scopes are subsystem names: `cli`, `daemon`,
  `velocity`, `vectors`, `focus`, `insights`, `privacy`.
- **Tests** live flat in `tests/`, one `test_<module>.py` per source module.
  Test fixtures construct events the way real ingestion produces them — in
  particular, `claude_project` carries a full cwd path, never a bare name.
- **Tunables** belong in `src/ambient/config.py` as dataclass fields:
  snake_case prefixed by detector name, `_ms` suffix for durations, with an
  inline comment giving the human-readable value.

## Privacy Contract

`docs/PRIVACY.md` is a binding contract, not advisory documentation. Any new
capture unit (anything that writes a new file under `~/.ambient/`, subscribes
to a new OS notification class, or hooks a new external system) must cite the
specific clauses it satisfies, by number, before it can merge. Some signal
classes (raw keystrokes, clipboard contents, full URL history) are permanently
closed. New Anthropic API call sites must be enumerated in clause 9.

## Reporting Issues

Include macOS version, Python version, and the relevant lines from
`~/.ambient/daemon/daemon.log`. Never paste the contents of
`~/.ambient/logs/` event files into an issue — they contain your raw command
history.
