import zlib
from collections import defaultdict
from dataclasses import dataclass

from ambient.capture.reader import Event
from ambient.config import Config


@dataclass
class RepeatedSequence:
    sequence: tuple[str, ...]
    count: int
    total_time_ms: int
    compression_gain: int  # count * len(sequence)


@dataclass
class CompressionFindings:
    sequences: list[RepeatedSequence]
    compression_ratio: float  # zlib ratio on full token stream


def _find_subsequences(
    commands: list[str],
    min_len: int,
    max_len: int,
    min_freq: int,
) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for window_size in range(min_len, max_len + 1):
        for i in range(len(commands) - window_size + 1):
            seq = tuple(commands[i : i + window_size])
            counts[seq] += 1
    return {seq: count for seq, count in counts.items() if count >= min_freq}


def _is_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    if len(short) >= len(long):
        return False
    for i in range(len(long) - len(short) + 1):
        if long[i : i + len(short)] == short:
            return True
    return False


def _dedup_subsequences(
    counts: dict[tuple[str, ...], int],
    ratio_threshold: float,
) -> dict[tuple[str, ...], int]:
    # Sort by length descending so we check longer sequences first
    sorted_seqs = sorted(counts.keys(), key=len, reverse=True)
    suppressed: set[tuple[str, ...]] = set()

    for i, short_seq in enumerate(sorted_seqs):
        if short_seq in suppressed:
            continue
        for long_seq in sorted_seqs:
            if long_seq in suppressed:
                continue
            if len(long_seq) <= len(short_seq):
                continue
            if _is_subsequence(short_seq, long_seq):
                # Suppress the shorter if the longer covers enough of its occurrences
                if counts[long_seq] / counts[short_seq] >= ratio_threshold:
                    suppressed.add(short_seq)
                    break

    return {seq: count for seq, count in counts.items() if seq not in suppressed}


def _compute_compression_ratio(commands: list[str]) -> float:
    if not commands:
        return 1.0
    raw = "\n".join(commands).encode("utf-8")
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)


def detect_compression(events: list[Event], config: Config) -> CompressionFindings:
    commands = [e.command for e in events]

    if len(commands) < config.min_sequence_length:
        return CompressionFindings(sequences=[], compression_ratio=1.0)

    # Find repeated subsequences
    counts = _find_subsequences(
        commands,
        min_len=config.min_sequence_length,
        max_len=config.max_sequence_length,
        min_freq=config.min_sequence_frequency,
    )

    # Deduplicate
    counts = _dedup_subsequences(counts, config.subsequence_dedup_ratio)

    # Build time index for total_time_ms calculation
    cmd_times = [e.duration_ms for e in events]

    sequences = []
    for seq, count in counts.items():
        seq_len = len(seq)
        # Estimate total time: find all matching windows and sum their durations
        total_time = 0
        for i in range(len(commands) - seq_len + 1):
            if tuple(commands[i : i + seq_len]) == seq:
                total_time += sum(cmd_times[i : i + seq_len])

        sequences.append(
            RepeatedSequence(
                sequence=seq,
                count=count,
                total_time_ms=total_time,
                compression_gain=count * seq_len,
            )
        )

    # Sort by compression gain descending
    sequences.sort(key=lambda s: s.compression_gain, reverse=True)

    compression_ratio = _compute_compression_ratio(commands)

    return CompressionFindings(
        sequences=sequences,
        compression_ratio=compression_ratio,
    )
