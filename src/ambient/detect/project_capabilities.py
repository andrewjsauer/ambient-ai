"""Project capability detection.

Probes a project's working directory for evidence of test, typecheck, and
dev-server capabilities. Used by verification.py to bucket fix sessions so
the report can distinguish "no tests run" from "no tests exist".

Capabilities are cached per-cwd within a single insights run. Call
clear_capability_cache() at the top of aggregate_coaching_data so a fresh
probe runs on every report (capabilities can change between runs).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Capabilities:
    has_tests: bool = False
    has_typecheck: bool = False
    has_dev_server: bool = False
    # Maps capability name -> short string describing the evidence.
    # Empty dict when no capabilities detected.
    evidence: dict[str, str] = field(default_factory=dict)


_capability_cache: dict[str, Capabilities] = {}


def clear_capability_cache() -> None:
    """Reset the per-run cache. Call once at the start of each insights run."""
    _capability_cache.clear()


def detect_capabilities(cwd: str | None) -> Capabilities:
    """Probe `cwd` for test, typecheck, and dev-server capability evidence.

    Walks the cwd looking for recognized project metadata files. Returns a
    Capabilities record whose `evidence` field cites the specific file/script
    that proved each capability, so misclassifications are debuggable.

    Returns an empty Capabilities for missing, unreadable, or inaccessible
    cwds; never raises. The honors-the-contract guarantee covers
    PermissionError on parent directories (mode 700 owned by another user,
    expired NFS ACLs) which `Path.is_dir()` would otherwise propagate.
    """
    if not cwd:
        return Capabilities()

    cached = _capability_cache.get(cwd)
    if cached is not None:
        return cached

    caps = Capabilities()
    root = Path(cwd)

    if not _safe_is_dir(root):
        _capability_cache[cwd] = caps
        return caps

    _probe_package_json(root, caps)
    _probe_pyproject(root, caps)
    _probe_cargo(root, caps)
    _probe_makefile(root, caps)
    _probe_tsconfig(root, caps)

    _capability_cache[cwd] = caps
    return caps


def _safe_is_file(path: Path) -> bool:
    """Path.is_file() that returns False instead of raising on permission/IO errors."""
    try:
        return path.is_file()
    except OSError as e:
        logger.warning("project_capabilities: is_file failed for %s: %s", path, e)
        return False


def _safe_is_dir(path: Path) -> bool:
    """Path.is_dir() that returns False instead of raising on permission/IO errors."""
    try:
        return path.is_dir()
    except OSError as e:
        logger.warning("project_capabilities: is_dir failed for %s: %s", path, e)
        return False


def _probe_package_json(root: Path, caps: Capabilities) -> None:
    pkg = root / "package.json"
    if not _safe_is_file(pkg):
        return
    try:
        data = json.loads(pkg.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("project_capabilities: malformed package.json at %s: %s", pkg, e)
        return

    scripts = data.get("scripts") or {}
    if not isinstance(scripts, dict):
        return

    if scripts.get("test") and not caps.has_tests:
        caps.has_tests = True
        caps.evidence["has_tests"] = f"package.json scripts.test={scripts['test']!r}"
    if scripts.get("dev") and not caps.has_dev_server:
        caps.has_dev_server = True
        caps.evidence["has_dev_server"] = f"package.json scripts.dev={scripts['dev']!r}"
    # Common typecheck script names. tsconfig.json probe also flips this.
    for name in ("typecheck", "type-check", "tsc"):
        if scripts.get(name) and not caps.has_typecheck:
            caps.has_typecheck = True
            caps.evidence["has_typecheck"] = f"package.json scripts.{name}={scripts[name]!r}"
            break


_PYTEST_SECTION_RE = re.compile(r"^\[tool\.pytest", re.MULTILINE)


def _probe_pyproject(root: Path, caps: Capabilities) -> None:
    pyproject = root / "pyproject.toml"
    has_pyproject = _safe_is_file(pyproject)
    if has_pyproject:
        try:
            text = pyproject.read_text()
        except OSError as e:
            logger.warning("project_capabilities: unreadable pyproject.toml at %s: %s",
                           pyproject, e)
            text = ""
        # Anchor at start-of-line so commented-out sections (`# [tool.pytest...`)
        # and doc-string mentions don't fire a false positive.
        if _PYTEST_SECTION_RE.search(text) and not caps.has_tests:
            caps.has_tests = True
            caps.evidence["has_tests"] = "pyproject.toml [tool.pytest...]"

    # A bare `tests/` directory is evidence of test capability only for
    # Python projects (gated by pyproject.toml or setup.py presence). Without
    # the gate this fires for any language with a `tests/` folder, mis-bucketing
    # JS/Go/etc. projects into `has_tests` and inflating their gap rates.
    if (
        not caps.has_tests
        and (has_pyproject or _safe_is_file(root / "setup.py"))
        and _safe_is_dir(root / "tests")
    ):
        caps.has_tests = True
        caps.evidence["has_tests"] = "tests/ directory present (Python project)"


def _probe_cargo(root: Path, caps: Capabilities) -> None:
    if _safe_is_file(root / "Cargo.toml") and not caps.has_tests:
        # Cargo always supports `cargo test`; presence of Cargo.toml is sufficient.
        caps.has_tests = True
        caps.evidence["has_tests"] = "Cargo.toml present (cargo test always available)"


_MAKE_TARGET_RE = re.compile(r"^([a-zA-Z0-9_.-]+)\s*:", re.MULTILINE)
_TEST_TARGET_NAMES = {"test", "tests", "check"}
_DEV_TARGET_NAMES = {"dev", "serve", "run", "start"}


def _probe_makefile(root: Path, caps: Capabilities) -> None:
    makefile = root / "Makefile"
    if not _safe_is_file(makefile):
        return
    try:
        text = makefile.read_text()
    except OSError as e:
        logger.warning("project_capabilities: unreadable Makefile at %s: %s", makefile, e)
        return

    targets = {m.group(1).lower() for m in _MAKE_TARGET_RE.finditer(text)}
    if not caps.has_tests:
        for name in _TEST_TARGET_NAMES:
            if name in targets:
                caps.has_tests = True
                caps.evidence["has_tests"] = f"Makefile target '{name}'"
                break
    if not caps.has_dev_server:
        for name in _DEV_TARGET_NAMES:
            if name in targets:
                caps.has_dev_server = True
                caps.evidence["has_dev_server"] = f"Makefile target '{name}'"
                break


def _probe_tsconfig(root: Path, caps: Capabilities) -> None:
    if _safe_is_file(root / "tsconfig.json") and not caps.has_typecheck:
        caps.has_typecheck = True
        caps.evidence["has_typecheck"] = "tsconfig.json present"
