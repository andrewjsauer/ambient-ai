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

    # API
    haiku_model: str = "claude-haiku-4-5"
    sonnet_model: str = "claude-sonnet-4-6"

    # Analysis
    default_window_minutes: int = 30

    # Daemon
    daemon_dir: Path = field(default=None)

    def __post_init__(self):
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"
        if self.analysis_dir is None:
            self.analysis_dir = self.base_dir / "analysis"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.daemon_dir is None:
            self.daemon_dir = self.base_dir / "daemon"

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
