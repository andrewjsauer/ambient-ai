import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DaemonState:
    last_analyzed_ts: int = 0  # epoch ms, exclusive lower bound
    last_summary_date: str = ""  # YYYY-MM-DD, last date a summary was generated
    last_calibration_date: str = ""  # YYYY-MM-DD
    events_since_calibration: int = 0

    @classmethod
    def load(cls, path: Path) -> "DaemonState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                last_analyzed_ts=data.get("last_analyzed_ts", 0),
                last_summary_date=data.get("last_summary_date", ""),
                last_calibration_date=data.get("last_calibration_date", ""),
                events_since_calibration=data.get("events_since_calibration", 0),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self), f, indent=2)
                f.write("\n")
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
