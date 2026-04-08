from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.projects import detect_project_allocation


def _cmd(ts, cwd="/projects/alpha", duration_ms=1000):
    return Event(
        ts_start=ts,
        ts_end=ts + duration_ms,
        duration_ms=duration_ms,
        command="echo hi",
        exit_code=0,
        cwd=cwd,
        tmux_pane=None,
        gap_ms=None,
    )


def _session(ts, project="/projects/beta", duration_ms=60000):
    return Event(
        ts_start=ts,
        ts_end=ts + duration_ms,
        duration_ms=duration_ms,
        command="claude: fix tests",
        exit_code=0,
        cwd=project,
        tmux_pane=None,
        gap_ms=None,
        type="claude_session",
        claude_project=project,
        claude_prompt_count=5,
    )


def test_multiple_projects():
    config = Config()
    events = [
        _cmd(1000, cwd="/projects/alpha", duration_ms=5000),
        _cmd(2000, cwd="/projects/alpha", duration_ms=3000),
        _cmd(3000, cwd="/projects/beta", duration_ms=2000),
        _session(4000, project="/projects/gamma", duration_ms=60000),
    ]

    result = detect_project_allocation(events, config)
    assert len(result.allocations) == 3

    by_name = {a.project: a for a in result.allocations}
    assert by_name["alpha"].total_ms == 8000
    assert by_name["alpha"].event_count == 2
    assert by_name["beta"].total_ms == 2000
    assert by_name["gamma"].total_ms == 60000
    assert by_name["gamma"].session_count == 1
    assert result.primary_project == "gamma"


def test_context_switches():
    config = Config()
    events = [
        _cmd(1000, cwd="/projects/alpha"),
        _cmd(2000, cwd="/projects/beta"),
        _cmd(3000, cwd="/projects/alpha"),
        _cmd(4000, cwd="/projects/alpha"),
    ]

    result = detect_project_allocation(events, config)
    assert result.context_switches == 2


def test_no_cwd_or_project():
    config = Config()
    events = [
        Event(ts_start=1000, ts_end=2000, duration_ms=1000, command="echo",
              exit_code=0, cwd="", tmux_pane=None, gap_ms=None),
    ]

    result = detect_project_allocation(events, config)
    assert result.allocations[0].project == "unknown"


def test_single_project():
    config = Config()
    events = [
        _cmd(1000, cwd="/projects/only"),
        _cmd(2000, cwd="/projects/only"),
    ]

    result = detect_project_allocation(events, config)
    assert result.context_switches == 0
    assert len(result.allocations) == 1


def test_empty_events():
    config = Config()
    result = detect_project_allocation([], config)
    assert result.allocations == []
    assert result.context_switches == 0
    assert result.primary_project == ""
