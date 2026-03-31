import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.compression import detect_compression, _dedup_subsequences


def _make_events(commands: list[str], duration_ms: int = 100) -> list[Event]:
    events = []
    ts = 1000
    for cmd in commands:
        events.append(
            Event(
                ts_start=ts,
                ts_end=ts + duration_ms,
                duration_ms=duration_ms,
                command=cmd,
                exit_code=0,
                cwd="/tmp",
                tmux_pane="%0",
                gap_ms=500 if events else None,
            )
        )
        ts += duration_ms + 500
    return events


@pytest.fixture
def config():
    return Config(min_sequence_frequency=2)  # Lower threshold for testing


def test_finds_repeated_sequence(config):
    # a,b,c repeated 3 times
    cmds = ["a", "b", "c"] * 3 + ["d"]
    events = _make_events(cmds)
    result = detect_compression(events, config)

    # Should find (a,b,c) with count 3
    abc = [s for s in result.sequences if s.sequence == ("a", "b", "c")]
    assert len(abc) == 1
    assert abc[0].count == 3
    assert abc[0].compression_gain == 9  # 3 * 3


def test_longer_sequences_rank_higher(config):
    # a,b repeated 4x and a,b,c repeated 4x
    cmds = ["a", "b", "c"] * 4
    events = _make_events(cmds)
    result = detect_compression(events, config)

    gains = {s.sequence: s.compression_gain for s in result.sequences}
    # (a,b,c) * 4 = gain 12 should beat (a,b) * 4 = gain 8 (if a,b survives dedup)
    if ("a", "b", "c") in gains:
        abc_gain = gains[("a", "b", "c")]
        for seq, gain in gains.items():
            if len(seq) < 3:
                assert abc_gain >= gain


def test_short_input_returns_empty(config):
    events = _make_events(["a"])
    result = detect_compression(events, config)
    assert result.sequences == []


def test_all_unique_returns_empty(config):
    events = _make_events(["a", "b", "c", "d", "e", "f"])
    result = detect_compression(events, config)
    assert result.sequences == []


def test_dedup_suppresses_when_long_covers_short():
    counts = {
        ("a", "b"): 3,
        ("a", "b", "c"): 3,  # covers 100% of (a,b) occurrences
    }
    result = _dedup_subsequences(counts, ratio_threshold=0.80)
    assert ("a", "b") not in result
    assert ("a", "b", "c") in result


def test_dedup_keeps_short_when_long_doesnt_cover():
    counts = {
        ("a", "b"): 10,
        ("a", "b", "c"): 2,  # only covers 20% of (a,b) = 2/10
    }
    result = _dedup_subsequences(counts, ratio_threshold=0.80)
    assert ("a", "b") in result
    assert ("a", "b", "c") in result


def test_dedup_boundary_ratio():
    # 67% < 80% threshold — keep both
    counts = {
        ("a", "b"): 3,
        ("a", "b", "c"): 2,  # 2/3 = 67%
    }
    result = _dedup_subsequences(counts, ratio_threshold=0.80)
    assert ("a", "b") in result

    # 100% >= 80% — suppress short
    counts2 = {
        ("a", "b"): 3,
        ("a", "b", "c"): 3,  # 3/3 = 100%
    }
    result2 = _dedup_subsequences(counts2, ratio_threshold=0.80)
    assert ("a", "b") not in result2


def test_compression_ratio_repetitive_vs_random(config):
    repetitive = _make_events(["git add .", "git commit -m wip", "git push"] * 10)
    random_cmds = _make_events([f"unique_cmd_{i}" for i in range(30)])

    rep_result = detect_compression(repetitive, config)
    rand_result = detect_compression(random_cmds, config)

    assert rep_result.compression_ratio < rand_result.compression_ratio


def test_realistic_workflow(config):
    cmds = (
        ["git status", "pytest tests/", "git add .", "git commit -m fix", "git push"] * 4
        + ["vim parser.py", "python run.py", "vim parser.py", "python run.py"] * 3
        + ["docker build .", "docker run app"]
    )
    events = _make_events(cmds)
    result = detect_compression(events, config)

    # Should find the git sequence and the vim/python cycle
    found_seqs = [s.sequence for s in result.sequences]
    assert len(result.sequences) > 0
    # The top result should have meaningful compression gain
    assert result.sequences[0].compression_gain > 4
