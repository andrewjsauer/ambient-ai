from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    base_dir: Path = field(default_factory=lambda: Path.home() / ".ambient")
    logs_dir: Path = field(default=None)
    analysis_dir: Path = field(default=None)
    models_dir: Path = field(default=None)

    # Shell hooks
    ignore_commands: list[str] = field(
        default_factory=lambda: ["ls", "cd", "clear", "pwd", "echo"]
    )
    session_boundary_ms: int = 600_000  # 10 minutes

    # Compression detector
    min_sequence_length: int = 2
    max_sequence_length: int = 8
    min_sequence_frequency: int = 3
    subsequence_dedup_ratio: float = 0.80

    # Pause classifier (GMM)
    gmm_n_components: int = 3
    gmm_min_samples: int = 60
    gmm_covariance_type: str = "diag"
    gmm_n_init: int = 10

    # Changepoint detector
    bucket_minutes: int = 5
    pelt_min_size: int = 10
    pelt_model: str = "l1"

    # Prompt pattern detector
    prompt_pattern_min_frequency: int = 3
    prompt_pattern_max_length: int = 4
    # Max wall-clock gap between adjacent sessions for cross-session n-gram linking.
    prompt_pattern_cross_session_max_gap_ms: int = 86_400_000  # 24 hours

    # Coaching / thrash detection
    thrash_score_threshold: float = 0.5
    thrash_min_prompts: int = 3
    # Minimum qualifying sessions before pooling thrash into averages.
    thrash_aggregate_min_n: int = 5

    # Resolution velocity
    velocity_idle_break_ms: int = 900_000  # 15 minutes
    velocity_min_chains: int = 5
    # Per-event ceilings on contribution to chain active_time_ms. Long-running
    # foreground processes (dev servers, watchers) should not dominate the
    # "active debugging time" metric even though they sit inside the chain.
    velocity_max_command_contribution_ms: int = 600_000  # 10 minutes
    velocity_max_session_contribution_ms: int = 3_600_000  # 60 minutes
    # Abandonment-reason classification (Unit 2): how many Read/Grep/ToolSearch
    # calls in a session with zero Edit/Write constitute "context rot".
    velocity_context_rot_min_tool_calls: int = 5

    # Verification-gap detector (Unit 1)
    verification_gap_window_ms: int = 300_000  # 5 minutes
    verification_min_fix_sessions: int = 10  # floor for emitting gap_rate
    verification_test_command_patterns: list[str] = field(default_factory=lambda: [
        "pytest",
        "npm test",
        "npm run test",
        "yarn test",
        "pnpm test",
        "bun test",
        "cargo test",
        "go test",
        "make test",
        "rake test",
        "mix test",
        "jest",
        "vitest",
        "deno test",
        "rspec",
    ])
    # Build/typecheck commands that count as verification for projects that
    # have no test target but do have a typecheck or build step. Used by the
    # project-aware verification-gap detector for the `has_typecheck` bucket.
    verification_typecheck_command_patterns: list[str] = field(default_factory=lambda: [
        "tsc",
        "pnpm typecheck",
        "pnpm type-check",
        "npm run typecheck",
        "npm run type-check",
        "yarn typecheck",
        "yarn type-check",
        "next build",
        "vite build",
        "tsup",
        "cargo check",
        "cargo build",
        "go build",
        "mypy",
        "pyright",
    ])

    # Git activity detector (v3 Unit 3): per-project commits read from
    # `git log` for the insights window. Cap protects against pathological
    # repos with thousands of commits in the window.
    git_activity_max_commits: int = 50

    # Stuck-loop trigger-prompt diagnostic (legacy Unit 3 from prior plan)
    coaching_vague_framing_patterns: list[str] = field(default_factory=lambda: [
        r"\bfigure out\b",
        r"\bfix (this|that|it)\b",
        r"\bdebug (this|that|it)\b",
        r"\bwhat.?s wrong\b",
        r"\bwhy (is|isn.?t)\b",
        r"\bunderstand why\b",
        r"\bhelp me\b",
        r"\bsomething.?s broken\b",
        r"\bnot working\b",
    ])

    # API
    haiku_model: str = "claude-haiku-4-5"
    sonnet_model: str = "claude-sonnet-4-6"

    # Analysis
    default_window_minutes: int = 30
    weekly_min_weeks: int = 2

    # Command-mix detector: per-project floor for inclusion in per_project map.
    # Below this floor, prompts still roll into the overall mix.
    command_mix_min_prompts: int = 10

    # Freeform-fraction detector: per-project floor for inclusion in per_project breakdown.
    freeform_fraction_min_prompts: int = 20

    # Project-ledger detector: per-project active-time floor (10 minutes default),
    # top-files and prompt caps, and per-prompt truncation for the Haiku summary.
    project_ledger_min_active_ms: int = 600_000  # 10 minutes
    project_ledger_top_files_n: int = 5
    project_ledger_summary_max_prompts: int = 30
    project_ledger_summary_truncate_chars: int = 500

    # v4 Phase 3: vector-aggregation detector tunables.
    # Pause label threshold for emitting a pause stop event (the GMM classifier
    # produces labels like "routine", "evaluating", "stuck"). Anything ≥ this
    # label in severity counts as a stop. "evaluating" is the conservative
    # default; calibrate against real data.
    vector_pause_min_label: str = "evaluating"
    # Drop a focus-change stop event if the previous focus-change was within
    # this many ms (alt-tab through 5 apps in 1s collapses to one stop).
    vector_focus_debounce_ms: int = 2000
    # Renderer caps: top-N longest vectors per project, top-N overall.
    vectors_per_project: int = 3
    longest_vectors_overall: int = 5

    # Slash-command taxonomy: per-user reclassification of custom commands.
    # Maps "/command" → category ("planning"|"execution"|"review"|"design"|"meta"|"other").
    # Built-in classifications cover the validated set from the inventory;
    # this field lets users add or reroute their own custom commands without
    # editing slash_taxonomy.py.
    slash_taxonomy_overrides: dict[str, str] = field(default_factory=dict)

    # Daemon
    daemon_dir: Path = field(default=None)
    claude_history_path: Path = field(default=None)
    claude_projects_dir: Path = field(default=None)

    # v4 Phase 2 Unit 7: NSWorkspace focus listener (separate launchd agent).
    # Off by default (cites docs/PRIVACY.md clause 6 — opt-in per signal class).
    # When enabled, app-activation events go to focus_events_path; the listener
    # daemon's stdout/stderr go to dedicated paths (separate from the daemon
    # log file so launchd doesn't interleave with our logging.FileHandler).
    focus_capture_enabled: bool = False
    focus_events_path: Path = field(default=None)
    focus_listener_log_path: Path = field(default=None)
    focus_listener_lock_path: Path = field(default=None)
    focus_listener_stdout_path: Path = field(default=None)
    focus_listener_stderr_path: Path = field(default=None)

    def __post_init__(self):
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"
        if self.analysis_dir is None:
            self.analysis_dir = self.base_dir / "analysis"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.daemon_dir is None:
            self.daemon_dir = self.base_dir / "daemon"
        if self.claude_history_path is None:
            self.claude_history_path = Path.home() / ".claude" / "history.jsonl"
        if self.claude_projects_dir is None:
            self.claude_projects_dir = Path.home() / ".claude" / "projects"
        if self.focus_events_path is None:
            self.focus_events_path = self.base_dir / "focus-events.jsonl"
        if self.focus_listener_log_path is None:
            self.focus_listener_log_path = self.base_dir / "focus-listener.log"
        if self.focus_listener_lock_path is None:
            self.focus_listener_lock_path = self.daemon_dir / "focus-listener.lock"
        if self.focus_listener_stdout_path is None:
            self.focus_listener_stdout_path = self.base_dir / "focus-listener-stdout.log"
        if self.focus_listener_stderr_path is None:
            self.focus_listener_stderr_path = self.base_dir / "focus-listener-stderr.log"

    def ensure_dirs(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.daemon_dir.mkdir(parents=True, exist_ok=True)

    def events_path(self, date_str: str) -> Path:
        return self.logs_dir / f"events-{date_str}.jsonl"

    def analysis_path(self, date_str: str) -> Path:
        return self.analysis_dir / f"analysis-{date_str}.jsonl"

    def summary_path(self, date_str: str) -> Path:
        return self.analysis_dir / f"summary-{date_str}.md"

    def weekly_summary_path(self, date_str: str) -> Path:
        return self.analysis_dir / f"weekly-{date_str}.md"

    def insights_path(self, date_str: str) -> Path:
        return self.base_dir / "insights" / f"insights-{date_str}.md"

    @property
    def recommendations_dir(self) -> Path:
        return self.base_dir / "recommendations"

    @property
    def gmm_model_path(self) -> Path:
        return self.models_dir / "gmm.joblib"

    @property
    def lock_path(self) -> Path:
        return self.daemon_dir / "daemon.lock"

    @property
    def state_path(self) -> Path:
        return self.daemon_dir / "state.json"

    @property
    def daemon_log_path(self) -> Path:
        return self.daemon_dir / "daemon.log"

    @property
    def dotenv_path(self) -> Path:
        return self.base_dir / ".env"

