# Privacy Policy — ambient-ai

**Status:** Active contract. Future capture units must verify against the rules below by clause number. Changing a "closed door" requires explicit user approval recorded in this document.
**Last reviewed:** 2026-06-12
**Scope:** All capture, storage, processing, and external transmission performed by ambient-ai on the user's local machine.

ambient-ai is a personal-tool behavioral monitor for the user's own development workflow. The data it captures is sensitive — command history, file paths, prompt text, focus events. The system grows by accumulating signals the user trusts, not by capturing everything technically possible. This document is the contract that governs that growth.

---

## Section 1 — Capture Rules

The following rules apply to every signal class ambient-ai captures, today or in the future.

1. **No clipboard contents.** The clipboard frequently holds passwords, tokens, and confidential text. ambient-ai never persists clipboard content. A future signal class may capture clipboard *event metadata* — booleans (a write happened), event timestamps, content lengths bucketed (small/medium/large) — but never the bytes themselves.

2. **No full URLs or browser history.** If browser-side capture is ever added, it records the *origin* (`example.com`) and optionally the first path segment for app-internal navigation. Query strings, fragments, full paths, search terms, and cookies are out of scope.

3. **No calendar event content.** A future calendar-overlap signal records a boolean "busy block active" and a coarse time-of-day class only. Event titles, attendees, descriptions, and recurrence rules are never persisted.

4. **No raw keystrokes.** System-wide keystroke capture (CGEventTap on macOS, equivalent APIs elsewhere) is permanently closed under this policy. A future cadence signal may compute keystroke-cadence histograms (median ms between keys, backspace ratio) but never persists individual keystrokes.

5. **Pre-redaction on draft text.** If a future signal class ever captures pre-Enter draft text (today: out of scope), every draft passes through a credential-shape regex pass — AWS keys, GitHub tokens, JWTs, hex secrets ≥32 chars, env-var-style `KEY=value` pairs — and matches are dropped before disk write. The redactor is the boundary between capture and storage; nothing un-redacted ever lands.

6. **All capture is opt-in per signal class.** ambient-ai never enables a new signal class by default. Each is gated behind an explicit CLI subcommand (`ambient focus-enable`, `ambient tmux-focus-enable`, etc.) that requires user invocation. Disabling a signal class is symmetric: `ambient focus-disable` removes the capture daemon and stops new writes. No "enable everything" toggle exists.

7. **Local-only by default.** All raw event data lives on the user's device under `~/.ambient/`. The Anthropic API is the only external boundary, and only *aggregated detector findings* cross it — never raw events, never raw prompts beyond a per-project ledger summarizer call (see clause 9). The user can run ambient-ai indefinitely with `ANTHROPIC_API_KEY` unset; only the LLM-narration features are gated by it.

8. **Quarantine for browser data.** If a browser extension is ever added (today: out of scope; deferred to a separate repo), it operates behind the extension boundary — communicating with the local daemon over a defined IPC surface, never reading other apps' SQLite stores or arbitrary cookies. The browser-extension signal class enters under its own opt-in toggle and its own privacy-rules amendment to this document.

9. **API-bound payload review.** The four existing LLM call-site modules — `present/narrator.py` (`narrate_batch` Haiku; `narrate_daily`/`narrate_weekly` Sonnet), `present/insights.py` (coaching report, Sonnet), `present/recommender.py` (skill and coaching drafts, Haiku), and `detect/project_ledger.py` (`_summarize`, Haiku) — send aggregated findings and a capped subset of user prompts. These call sites are the entire external surface. Every new external call site must be enumerated here before merge.

---

## Section 2 — Closed Doors

The following signal classes are **permanently excluded** under this policy as written. Adding any one requires amending this document with explicit user approval recorded inline.

| Closed door | Why |
|---|---|
| **CGEventTap / system-wide keystroke capture** | Captures input across all apps including password managers. Privacy cost catastrophically exceeds any conceivable signal value. |
| **Full URL / browser history** | Search terms, query strings, and tokens-in-URL leak too much. Origin-only is the upper bound. |
| **Clipboard contents** | Passwords, tokens, confidential text. Booleans and lengths only. |
| **Calendar event titles, attendees, descriptions** | Meeting names and rosters are organizational PII that ambient-ai has no business persisting. |
| **Reading other apps' SQLite stores** | ambient-ai never opens another application's private data files (Chrome's history, Slack's local DB, Notes' store, etc.). |
| **Network packet capture** | Out of scope; would expose the contents of every TLS-decrypted browser session. |
| **Raw memory snapshots / keychain access** | Out of scope. |
| **Persistent cross-device sync of raw events** | Raw events stay on the device. Only aggregated insights leave, and only via the existing API boundary. |

---

## Section 3 — Verification Expectations

Every new capture unit (anything that writes to a new file under `~/.ambient/`, subscribes to a new OS notification class, or hooks a new external system) must include a **verification step** in its plan that cites specific clauses of this document by number.

Example (from the plan unit that introduced the NSWorkspace app-activation listener):

> **Verification:** … Verification cites `docs/PRIVACY.md` clauses 6 and 7 by number.

The verification step demonstrates that:

- The signal class is opt-in (clause 6).
- The captured payload contains no fields named in the closed-doors table (clauses in Section 2).
- The data path stays local-only or, if it crosses the API boundary, the cross-boundary payload is enumerated in clause 9.

A capture unit whose verification step does not cite at least one specific clause is incomplete and must not merge.

---

## Section 4 — Storage and Retention

- **Location:** All captured data lives under `~/.ambient/` on the user's device. Subdirectories include `logs/` (shell + claude session events), `analysis/` (detector findings), `daemon/` (state, lock, log), `models/` (GMM classifier), and (Phase 2 onward) `focus-events.jsonl`.
- **Default retention:** 30 days for raw event data. Older files are eligible for log rotation. Aggregated weekly summaries persist longer (they are dramatically smaller and contain no raw events).
- **User control:** retention is configurable per signal class via `Config` fields. The user may delete `~/.ambient/` at any time without breaking the system; ambient-ai recreates the directory structure on next run.
- **Backups:** ambient-ai never automatically copies `~/.ambient/` off-device. If the user's home backup tool (Time Machine, Arq, etc.) includes it, that is the user's choice.

---

## Section 5 — Signal Classes and Current Status

| Signal class | Status | Default | Captured fields | Privacy notes |
|---|---|---|---|---|
| Shell command + exit code | Enabled | On | command text, cwd, exit code, timestamps | Existing zsh-hooks capture; clauses 1, 2, 7 apply. |
| Claude Code session metadata | Enabled | On (read of existing JSONL) | prompts, tool calls, file paths, session ids, ran-test / ran-typecheck / verification-resolved booleans | Read-only over `~/.claude/projects/`. clauses 1, 2, 7 apply (no clipboard, no URLs except as the user typed them in prompts; derived booleans stay local). Three booleans are derived from session Bash tool calls by structurally classifying each command's program: **ran-test** / **ran-typecheck** (did the session run a test or typecheck/build command) and **verification-resolved** (did a test fail then later pass in-session — a red→green fix loop). The command text and exit details are classified and discarded — only the booleans are stored, so no credential-bearing content is persisted (clauses 1, 5). |
| Pause / idle gaps | Enabled | On | gap durations between commands | Derived; no new payload. |
| App-activation events (NSWorkspace) | Phase 2 Unit 7 | **Opt-in (off by default)** | bundle id, localized app name, PID, timestamp | clauses 6, 7. **Never** window title, document path, or any field named in Section 2's closed-doors table. |
| tmux pane/window focus | Phase 2 Unit 8 | **Opt-in (off by default)** | hook name, pane id, window index, session name, timestamp | clause 7. **Never** `pane_current_command`, `pane_current_path`, or `pane_title`. |
| Keystroke cadence | **CLOSED** | n/a | n/a | Permanently excluded (clause 4) until a separate privacy-rules amendment. |
| Pre-Enter draft text | **CLOSED** | n/a | n/a | Permanently excluded until clause 5 redactor is implemented and reviewed. |
| Clipboard contents | **PERMANENTLY CLOSED** | n/a | n/a | Section 2 closed door. Requires document amendment. |
| Browser history | **CLOSED, deferred** | n/a | n/a | Section 2 closed door. May enter under a future browser-extension boundary (clause 8) with its own opt-in. |
| System-wide event taps (CGEventTap) | **PERMANENTLY CLOSED** | n/a | n/a | Section 2 closed door. |

---

## Section 6 — Amendment Process

To open a closed door or add a signal class beyond the current table:

1. Draft a plan unit that names the new signal class, its captured fields, and verification steps citing this document.
2. Update Section 5 with a new row including the rationale for opening the door.
3. Update Section 2 with a corresponding strikethrough or removal of the closed-door entry (with a dated note explaining the change).
4. Record the user's explicit approval inline in this document — date and the approving reasoning.
5. Reference this document section from the plan's verification step.

A capture change that does not include all five steps is out of scope.

---

## Section 7 — References

- Existing call sites: `src/ambient/present/api.py` (single Anthropic API entry point), `src/ambient/present/narrator.py` (batch Haiku + daily/weekly Sonnet), `src/ambient/present/insights.py` (coaching report, Sonnet), `src/ambient/present/recommender.py` (skill/coaching drafts, Haiku), `src/ambient/detect/project_ledger.py` (per-project Haiku summarizer).
