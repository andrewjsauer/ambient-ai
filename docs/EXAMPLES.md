# Example output

Every block below is produced by `scripts/gen_examples.py` from **synthetic** data — invented project names, prompts, and paths. No real `~/.ambient` or `~/.claude` data is read or committed. Regenerate with:

```bash
python scripts/gen_examples.py
```

## `ambient status`

Daemon health, today's activity, and what to run next.

```
Ambient · 2026-06-13
──────────────────────────────────────────────────
Daemon       running · last tick 14:51 · lock free · GMM calibrated
Today        43 events · 14 claude session(s) · last activity 13:38
Projects     infra (42m), web-app (36m), payments-api (29m)
Summary      today ✓ · latest 2026-06-13 · pending recs 2
Try          ambient review · ambient insights
```

## `ambient insights`

The coaching report: resolution velocity, stuck patterns, repeated prompts, and verification gaps.

```
Ambient Insights — 2026-06-06 to 2026-06-13

Worked on:            [infra] 42min · 3 session(s)
Worked on:            [web-app] 36min · 6 session(s)
Worked on:            [payments-api] 29min · 3 session(s)
Resolution velocity:  10.4 min avg (7 resolved)
Stuck episodes:       3
Thrash score:         0.38 avg
Top repeated prompt:  x4 "run the linter and fix everything"
Top command sequence: x3 git add -A -> git commit -m wip -> git push
Pending recs:         2
Verification gaps (tests): 4/11 fixes (36%)
Top trigger prompt:   "the state lock won't release"

Top finding: infra — 3 stuck episodes (42 min total)
```

## `ambient projects --window 2880`

Per-project time allocation and context switches.

```
=== Project Allocation (last 2880 minutes, 43 events) ===

PROJECT                         TIME      %   SESSIONS   EVENTS
---------------------------------------------------------------
infra                            42m    33%          3        6
web-app                          36m    28%          6       22
payments-api                     29m    23%          3        9
data-pipeline                    20m    16%          2        6

Context switches: 4
Primary project: infra
```

## `ambient stats --window 2880`

Raw algorithmic detector output — no LLM involved.

```
=== Stats for last 2880 minutes (43 events) ===

COMPRESSION:
  Compression ratio: 0.314
  git add -A -> git commit -m wip -> git push -> git add -A -> git commit -m wip -> git push (x3, gain=18)
  git add -A -> git commit -m wip -> git push (x4, gain=12)
  pytest -> pytest (x3, gain=6)
  terraform apply -> claude: the state lock won't release (x3, gain=6)
  claude: run the linter and fix everything -> claude: run the linter and fix everything (x3, gain=6)

PAUSE CLASSIFICATION:
  routine: 17/29 (59%)
  evaluating: 8/29 (28%)
  stuck: 4/29 (14%)
  Top stuck episodes:
    180000ms after 'claude: dedupe the nightly ingest rows'
    180000ms after 'git add -A'
    75000ms after 'pytest'

WORKFLOW RHYTHM (full day):
  225min | 1.0 cmd/5min | low-rate, test-focused

CLAUDE CODE SESSIONS (14 in window):
  Total time: 126 min
  14min | /tmp/ambient-demo-projects/payments-api | claude: fix the failing charge-refund test
  9min | /tmp/ambient-demo-projects/payments-api | claude: why does the webhook retry loop
  6min | /tmp/ambient-demo-projects/payments-api | claude: rounding is off on invoice totals
  12min | /tmp/ambient-demo-projects/web-app | claude: the checkout button fires twice
  11min | /tmp/ambient-demo-projects/web-app | claude: cart total wrong after coupon removal
  13min | /tmp/ambient-demo-projects/data-pipeline | claude: dedupe the nightly ingest rows
  7min | /tmp/ambient-demo-projects/data-pipeline | claude: null dates crash the loader
  14min | /tmp/ambient-demo-projects/infra | claude: the state lock won't release
  14min | /tmp/ambient-demo-projects/infra | claude: the state lock won't release
  14min | /tmp/ambient-demo-projects/infra | claude: the state lock won't release
  3min | /tmp/ambient-demo-projects/web-app | claude: run the linter and fix everything
  3min | /tmp/ambient-demo-projects/web-app | claude: run the linter and fix everything
  3min | /tmp/ambient-demo-projects/web-app | claude: run the linter and fix everything
  3min | /tmp/ambient-demo-projects/web-app | claude: run the linter and fix everything
```

## `ambient recommendations`

Installable skill / alias drafts staged from your patterns.

```
ID                             TYPE       TITLE
--------------------------------------------------------------------------------
alias-git-wip                  alias      Alias: gwip = git add -A && git commit -m wip && git push
skill-add-regression-test      skill      Skill: add a regression test for the failing case
```
