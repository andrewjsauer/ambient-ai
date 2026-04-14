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

    # Coaching / thrash detection
    thrash_score_threshold: float = 0.5
    thrash_min_prompts: int = 3
    # Minimum qualifying sessions before pooling thrash into averages.
    thrash_aggregate_min_n: int = 5

    # Resolution velocity
    velocity_idle_break_ms: int = 900_000  # 15 minutes
    velocity_min_chains: int = 5

    # API
    haiku_model: str = "claude-haiku-4-5"
    sonnet_model: str = "claude-sonnet-4-6"

    # Analysis
    default_window_minutes: int = 30
    weekly_min_weeks: int = 2

    # Daemon
    daemon_dir: Path = field(default=None)
    claude_history_path: Path = field(default=None)
    claude_projects_dir: Path = field(default=None)

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

