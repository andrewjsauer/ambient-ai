"""Tests for the git activity detector."""

import os
import shutil
import subprocess
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.git_activity import (
    GitActivityFindings,
    GitCommit,
    _parse_log_output,
    _parse_shortstat,
    detect_git_activity,
)

# Skip the integration tests if git isn't on PATH (CI minimal containers,
# Docker images without git installed). The pure-Python parser tests still run.
requires_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git not installed; integration tests require it",
)

# Isolate from machine-level git config so commit.gpgsign or aliases set
# in /etc/gitconfig can't interfere with test commits.
_GIT_ENV_NO_SYSTEM = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}


def _config(**overrides):
    return Config(**overrides)


def _cmd_event(cwd, ts_start=10_000):
    return Event(
        ts_start=ts_start,
        ts_end=ts_start + 1000,
        duration_ms=1000,
        command="ls",
        exit_code=0,
        cwd=cwd,
        tmux_pane=None,
        gap_ms=None,
        type="command",
    )


@pytest.fixture
def empty_repo(tmp_path):
    """A fresh git repo with no commits — useful for testing zero-commit windows."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)],
                   check=True, env=_GIT_ENV_NO_SYSTEM)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True, env=_GIT_ENV_NO_SYSTEM,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True, env=_GIT_ENV_NO_SYSTEM,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "commit.gpgsign", "false"],
        check=True, env=_GIT_ENV_NO_SYSTEM,
    )
    return tmp_path


def _commit(repo, filename, content, message):
    """Helper: write a file and commit it."""
    (repo / filename).write_text(content)
    subprocess.run(["git", "-C", str(repo), "add", filename],
                   check=True, env=_GIT_ENV_NO_SYSTEM)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", message],
        check=True, env=_GIT_ENV_NO_SYSTEM,
    )


class TestParseShortstat:
    def test_full_line(self):
        line = " 2 files changed, 14 insertions(+), 3 deletions(-)"
        assert _parse_shortstat(line) == (2, 14, 3)

    def test_insertions_only(self):
        line = " 1 file changed, 5 insertions(+)"
        assert _parse_shortstat(line) == (1, 5, 0)

    def test_deletions_only(self):
        line = " 1 file changed, 5 deletions(-)"
        assert _parse_shortstat(line) == (1, 0, 5)

    def test_garbage_line_returns_zeros(self):
        assert _parse_shortstat("not a stat line at all") == (0, 0, 0)

    def test_empty_line(self):
        assert _parse_shortstat("") == (0, 0, 0)


class TestParseLogOutput:
    def test_two_commits(self):
        output = (
            "abc1234567|2026-04-25T10:00:00-04:00|Alice|first commit\n"
            "\n"
            " 1 file changed, 5 insertions(+)\n"
            "\n"
            "def4567890|2026-04-25T11:00:00-04:00|Bob|second commit\n"
            "\n"
            " 2 files changed, 10 insertions(+), 1 deletion(-)\n"
        )
        commits = _parse_log_output(output)
        assert len(commits) == 2
        assert commits[0].sha == "abc1234567"
        assert commits[0].author == "Alice"
        assert commits[0].subject == "first commit"
        assert commits[0].insertions == 5
        assert commits[0].deletions == 0
        assert commits[1].subject == "second commit"
        assert commits[1].files_changed == 2
        assert commits[1].insertions == 10
        assert commits[1].deletions == 1

    def test_subject_with_pipe_character_preserved(self):
        """Subjects can legally contain |. Our split limit must not eat them."""
        output = "abc1234|2026-04-25T10:00:00-04:00|Alice|fix: handle a|b|c case\n"
        commits = _parse_log_output(output)
        assert len(commits) == 1
        assert commits[0].subject == "fix: handle a|b|c case"

    def test_header_without_shortstat_still_records_commit(self):
        """A commit with no file changes (e.g., empty commit) should still appear."""
        output = "abc1234|2026-04-25T10:00:00-04:00|Alice|empty commit\n"
        commits = _parse_log_output(output)
        assert len(commits) == 1
        assert commits[0].files_changed == 0

    def test_empty_output(self):
        assert _parse_log_output("") == []


@requires_git
class TestDetectGitActivityIntegration:
    """Real subprocess calls against tmp git repos."""

    def test_zero_commits_in_window(self, empty_repo):
        events = [_cmd_event(str(empty_repo))]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        result = detect_git_activity(events, start, end, _config())
        assert result == GitActivityFindings()

    def test_one_commit_in_window(self, empty_repo):
        _commit(empty_repo, "a.py", "x = 1\n", "feat: add x")
        events = [_cmd_event(str(empty_repo))]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)
        result = detect_git_activity(events, start, end, _config())
        assert result.total_commits == 1
        assert result.total_lines_changed >= 1
        project = empty_repo.name
        assert project in result.by_project
        assert result.by_project[project][0].subject == "feat: add x"

    def test_three_commits_sorted_newest_first(self, empty_repo):
        _commit(empty_repo, "a.py", "x = 1\n", "feat: a")
        _commit(empty_repo, "b.py", "y = 2\n", "feat: b")
        _commit(empty_repo, "c.py", "z = 3\n", "feat: c")
        events = [_cmd_event(str(empty_repo))]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)
        result = detect_git_activity(events, start, end, _config())
        assert result.total_commits == 3
        project = empty_repo.name
        subjects = [c.subject for c in result.by_project[project]]
        # git log --no-merges defaults to newest-first
        assert subjects == ["feat: c", "feat: b", "feat: a"]

    def test_commits_outside_window_excluded(self, empty_repo):
        _commit(empty_repo, "a.py", "x = 1\n", "feat: out-of-window")
        events = [_cmd_event(str(empty_repo))]
        # Window ends before the commit was made
        start = datetime.now() - timedelta(days=14)
        end = datetime.now() - timedelta(days=7)
        result = detect_git_activity(events, start, end, _config())
        assert result.total_commits == 0

    def test_max_commits_cap(self, empty_repo):
        for i in range(15):
            _commit(empty_repo, f"f{i}.py", f"x = {i}\n", f"feat: file {i}")
        events = [_cmd_event(str(empty_repo))]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)
        config = _config(git_activity_max_commits=5)
        result = detect_git_activity(events, start, end, config)
        project = empty_repo.name
        assert len(result.by_project[project]) == 5
        assert result.total_commits == 5

    def test_non_git_cwd_silently_skipped(self, tmp_path):
        events = [_cmd_event(str(tmp_path))]  # plain dir, not a git repo
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        result = detect_git_activity(events, start, end, _config())
        assert result == GitActivityFindings()

    def test_multiple_cwds_in_same_repo_dedupe_to_one_root(self, empty_repo):
        _commit(empty_repo, "a.py", "x = 1\n", "feat: a")
        # Subdirectory of the repo — `git rev-parse --show-toplevel` should
        # collapse both cwds to the same root, so we get one project not two.
        sub = empty_repo / "src"
        sub.mkdir()
        events = [
            _cmd_event(str(empty_repo)),
            _cmd_event(str(sub), ts_start=20_000),
        ]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)
        result = detect_git_activity(events, start, end, _config())
        assert len(result.by_project) == 1

    def test_two_separate_repos_appear_separately(self, tmp_path):
        repo_a = tmp_path / "alpha"
        repo_b = tmp_path / "beta"
        for repo in (repo_a, repo_b):
            repo.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "t@e.com"], check=True
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "T"], check=True
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "commit.gpgsign", "false"], check=True
            )
        _commit(repo_a, "a.py", "x = 1\n", "alpha: first")
        _commit(repo_b, "b.py", "y = 2\n", "beta: first")
        events = [_cmd_event(str(repo_a)), _cmd_event(str(repo_b), ts_start=20_000)]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)
        result = detect_git_activity(events, start, end, _config())
        assert set(result.by_project.keys()) == {"alpha", "beta"}
        assert result.total_commits == 2

    def test_empty_events_returns_empty_findings(self):
        result = detect_git_activity([], datetime.now() - timedelta(days=7),
                                     datetime.now(), _config())
        assert result == GitActivityFindings()

    def test_events_without_cwd_skipped(self):
        events = [
            Event(ts_start=10_000, ts_end=11_000, duration_ms=1000,
                  command="x", exit_code=0, cwd="", tmux_pane=None, gap_ms=None,
                  type="command"),
        ]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        result = detect_git_activity(events, start, end, _config())
        assert result == GitActivityFindings()


class TestSubprocessFailureHandling:
    def test_rev_parse_timeout_returns_no_findings(self, tmp_path):
        events = [_cmd_event(str(tmp_path))]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()

        def raising_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=a, timeout=5)

        with patch("ambient.detect.git_activity.subprocess.run", side_effect=raising_run):
            result = detect_git_activity(events, start, end, _config())
        assert result == GitActivityFindings()

    def test_log_timeout_skips_project(self, empty_repo):
        _commit(empty_repo, "a.py", "x = 1\n", "feat: a")
        events = [_cmd_event(str(empty_repo))]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)

        from ambient.detect import git_activity as ga
        original = subprocess.run

        def selective_timeout(*args, **kw):
            # Only the `log` call should time out; let `rev-parse` succeed
            cmd = args[0] if args else kw.get("args", [])
            if isinstance(cmd, list) and "log" in cmd:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)
            return original(*args, **kw)

        with patch.object(ga.subprocess, "run", side_effect=selective_timeout):
            result = detect_git_activity(events, start, end, _config())
        # rev-parse succeeded so the root was discovered, but log failed —
        # project gets no entry rather than crashing.
        assert result.total_commits == 0


@requires_git
class TestProjectName:
    def test_basename_used_as_project_label(self, tmp_path):
        repo = tmp_path / "my-cool-project"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)],
                       check=True, env=_GIT_ENV_NO_SYSTEM)
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.email", "t@e.com"],
            check=True, env=_GIT_ENV_NO_SYSTEM,
        )
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.name", "T"],
            check=True, env=_GIT_ENV_NO_SYSTEM,
        )
        subprocess.run(
            ["git", "-C", str(repo), "config", "commit.gpgsign", "false"],
            check=True, env=_GIT_ENV_NO_SYSTEM,
        )
        _commit(repo, "a.py", "x = 1\n", "feat: a")
        events = [_cmd_event(str(repo))]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)
        result = detect_git_activity(events, start, end, _config())
        assert "my-cool-project" in result.by_project

    def test_same_basename_collision_keeps_both_repos_distinct(self, tmp_path):
        """Two repos with identical basenames produce two distinct labels —
        no merge, no overcounting. Replaces v1's broken merge-sort path."""
        for parent in ("work", "personal"):
            repo = tmp_path / parent / "auth"
            repo.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", "-b", "main", str(repo)],
                           check=True, env=_GIT_ENV_NO_SYSTEM)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "t@e.com"],
                check=True, env=_GIT_ENV_NO_SYSTEM,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "T"],
                check=True, env=_GIT_ENV_NO_SYSTEM,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "commit.gpgsign", "false"],
                check=True, env=_GIT_ENV_NO_SYSTEM,
            )
            _commit(repo, f"{parent}.py", "x = 1\n", f"feat: {parent} auth")
        events = [
            _cmd_event(str(tmp_path / "work" / "auth")),
            _cmd_event(str(tmp_path / "personal" / "auth"), ts_start=20_000),
        ]
        start = datetime.now() - timedelta(days=7)
        end = datetime.now() + timedelta(minutes=1)
        result = detect_git_activity(events, start, end, _config())
        # Both repos should appear as distinct entries, not merged
        assert len(result.by_project) == 2
        # total_commits matches sum of per-project lists (no overcounting)
        assert result.total_commits == sum(len(v) for v in result.by_project.values())
        assert result.total_commits == 2
        # Labels disambiguated by parent dir
        labels = list(result.by_project.keys())
        assert "auth" in labels
        assert any("work" in lbl or "personal" in lbl for lbl in labels)


@requires_git
class TestRevParseDeduplication:
    def test_many_cwds_under_one_repo_call_rev_parse_once(self, empty_repo):
        """N cwds under the same repo should not invoke `git rev-parse` N times.
        The Python parent-walk cache short-circuits after the first."""
        _commit(empty_repo, "a.py", "x = 1\n", "feat: a")
        sub_a = empty_repo / "src"
        sub_b = empty_repo / "tests"
        sub_a.mkdir()
        sub_b.mkdir()
        events = [
            _cmd_event(str(empty_repo)),
            _cmd_event(str(sub_a), ts_start=20_000),
            _cmd_event(str(sub_b), ts_start=30_000),
        ]
        from ambient.detect import git_activity as ga
        call_count = {"n": 0}
        original = ga._git_root

        def counting_root(cwd):
            call_count["n"] += 1
            return original(cwd)

        with patch.object(ga, "_git_root", side_effect=counting_root):
            start = datetime.now() - timedelta(days=7)
            end = datetime.now() + timedelta(minutes=1)
            result = detect_git_activity(events, start, end, _config())

        # First cwd misses the cache and calls _git_root; the next two
        # find their root via the parent-walk and skip the subprocess.
        assert call_count["n"] == 1
        assert len(result.by_project) == 1


class TestSafeCwdGuard:
    def test_cwd_starting_with_dash_is_rejected(self):
        """A cwd like '--exec-path=/tmp/evil' must not be passed to git -C
        as it would be interpreted as a global flag."""
        events = [_cmd_event("--exec-path=/tmp/evil")]
        # Should not raise, should not invoke git, should return empty
        result = detect_git_activity(
            events,
            datetime.now() - timedelta(days=7),
            datetime.now(),
            _config(),
        )
        assert result == GitActivityFindings()

    def test_relative_cwd_is_rejected(self):
        """Defensively reject anything not absolute."""
        events = [_cmd_event("relative/path")]
        result = detect_git_activity(
            events,
            datetime.now() - timedelta(days=7),
            datetime.now(),
            _config(),
        )
        assert result == GitActivityFindings()


@requires_git
class TestNonzeroSubprocessExit:
    def test_log_nonzero_exit_skips_project(self, empty_repo):
        """A git log returning nonzero exit (e.g., corrupted history) must
        not crash the run; the project is just dropped."""
        from ambient.detect import git_activity as ga
        _commit(empty_repo, "a.py", "x = 1\n", "feat: a")
        events = [_cmd_event(str(empty_repo))]
        original = subprocess.run

        def selective_failure(*args, **kw):
            cmd = args[0] if args else kw.get("args", [])
            if isinstance(cmd, list) and "log" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=128, stdout="", stderr="fatal: bad object",
                )
            return original(*args, **kw)

        with patch.object(ga.subprocess, "run", side_effect=selective_failure):
            result = detect_git_activity(
                events,
                datetime.now() - timedelta(days=7),
                datetime.now() + timedelta(minutes=1),
                _config(),
            )
        assert result.total_commits == 0


class TestUnicodeRobustness:
    def test_non_utf8_subject_does_not_crash(self):
        """A commit subject containing non-UTF-8 bytes (legacy Latin-1 repo)
        must be tolerated. errors='replace' on subprocess.run keeps the
        decode safe; the subject lands with a replacement char."""
        # Simulate via direct parser test — the encoding behavior is set on
        # subprocess.run, so the parser sees pre-decoded text. This test
        # confirms the parser tolerates the replacement character.
        output = "abc1234567|2026-04-25T10:00:00-04:00|Alice|F\ufffdge Datei hinzu\n"
        commits = _parse_log_output(output)
        assert len(commits) == 1
        assert commits[0].sha == "abc1234567"
        assert "\ufffd" in commits[0].subject  # replacement char preserved


class TestParserDiscriminator:
    def test_shortstat_with_pipes_in_filename_not_misclassified_as_header(self):
        """A shortstat line that incidentally contains pipes shouldn't be
        treated as a commit header by the SHA-prefix discriminator."""
        # Pathological: imagine a shortstat where filename had pipes (rare,
        # but possible if git ever changes format). Our SHA hex check
        # rejects it.
        output = (
            "abc1234567|2026-04-25T10:00:00-04:00|Alice|first\n"
            " 1 file changed, 5 insertions(+), 3 deletions(-)\n"
            "weird|line|with|pipes|but|not|sha|prefix\n"
        )
        commits = _parse_log_output(output)
        # Only one commit; the pipe-heavy line is ignored
        assert len(commits) == 1
        assert commits[0].subject == "first"
