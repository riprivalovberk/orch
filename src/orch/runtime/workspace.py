"""Git workspace for one target repo: a branch per objective, a worktree per work task.

    main ──► obj/<root>            the objective's integration branch (own worktree)
               ├─► task/<task id>  one per test/implement task, cut from obj/<root> when
               └─► task/<task id>  the task is first claimed; merged back once it is DONE

Agents in parallel never share a directory. PAR-001 keeps parallel tasks off the same
files, so merges are clean. Once the objective is approved, obj/<root> merges into main.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_ID = ["-c", "user.name=orch", "-c", "user.email=orch@localhost", "-c", "commit.gpgsign=false"]


class GitError(RuntimeError):
    pass


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(
        ["git", *GIT_ID, *args], cwd=cwd, capture_output=True, text=True, timeout=120
    )
    if check and p.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {cwd}:\n{p.stderr or p.stdout}")
    return p


@dataclass(frozen=True)
class Merge:
    ok: bool
    detail: str = ""


class Workspace:
    def __init__(self, repo: Path, worktrees: Path, artifacts: Path | None = None) -> None:
        self.repo = repo.resolve()
        self.worktrees = worktrees.resolve()
        # Files agents hand in that are not code (reports, notes), stored by content hash.
        self.artifacts = (artifacts or self.worktrees.parent / "artifacts").resolve()

    @classmethod
    def create(cls, repo: Path, worktrees: Path, *, template: Path | None = None) -> Workspace:
        """Open a target repo, making it from a template folder (git init + commit) if new."""
        if not (repo / ".git").exists():
            if template is None:
                raise GitError(f"{repo} is not a git repo and no template was given")
            shutil.copytree(template, repo, dirs_exist_ok=True, ignore=_ignore)
            git(repo, "init", "-q", "-b", "main")
            git(repo, "add", "-A")
            git(repo, "commit", "-q", "-m", "Initial import")
        worktrees.mkdir(parents=True, exist_ok=True)
        return cls(repo, worktrees)

    def store_artifact(self, data: bytes) -> tuple[str, Path]:
        """Save bytes under their sha256; the same content is stored once."""
        sha = hashlib.sha256(data).hexdigest()
        path = self.artifacts / sha[:2] / sha
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return sha, path

    # ------------------------------------------------------------------ names

    @staticmethod
    def objective_branch(root_id: str) -> str:
        return f"obj/{root_id}"

    @staticmethod
    def task_branch(task_id: str) -> str:
        return f"task/{task_id}"

    def objective_dir(self, root_id: str) -> Path:
        return self.worktrees / root_id

    def task_dir(self, task_id: str) -> Path:
        return self.worktrees / task_id

    def _has_branch(self, name: str) -> bool:
        return (
            git(
                self.repo, "rev-parse", "--verify", "-q", f"refs/heads/{name}", check=False
            ).returncode
            == 0
        )

    def head(self, cwd: Path, ref: str = "HEAD") -> str:
        return git(cwd, "rev-parse", ref).stdout.strip()

    # ------------------------------------------------------------------ objective

    def start_objective(self, root_id: str) -> Path:
        """The objective's branch and worktree, made from main the first time."""
        d = self.objective_dir(root_id)
        if not d.exists():
            branch = self.objective_branch(root_id)
            if self._has_branch(branch):
                git(self.repo, "worktree", "add", "-q", str(d), branch)
            else:
                git(self.repo, "worktree", "add", "-q", "-b", branch, str(d), "main")
        return d

    def catch_up(self, root_id: str) -> Merge | None:
        """Bring main into the objective if main moved since it was built (another objective
        merged first). None when main has not moved; on a conflict the branch is left as is."""
        d = self.start_objective(root_id)
        if git(d, "merge-base", "--is-ancestor", "main", "HEAD", check=False).returncode == 0:
            return None
        p = git(d, "merge", "--no-edit", "-q", "main", check=False)
        if p.returncode != 0:
            git(d, "merge", "--abort", check=False)
            return Merge(False, p.stdout + p.stderr)
        return Merge(True, self.head(d))

    def undo_catch_up(self, root_id: str) -> None:
        git(self.objective_dir(root_id), "reset", "--hard", "-q", "ORIG_HEAD")

    def finish_objective(self, root_id: str) -> Merge:
        """Approved: merge the objective into main (the repo's own checkout)."""
        branch = self.objective_branch(root_id)
        p = git(self.repo, "merge", "--no-ff", "-q", "-m", f"Merge {branch}", branch, check=False)
        if p.returncode != 0:
            git(self.repo, "merge", "--abort", check=False)
            return Merge(False, p.stdout + p.stderr)
        return Merge(True, self.head(self.repo))

    # ------------------------------------------------------------------ tasks

    def open_task(self, root_id: str, task_id: str) -> Path:
        """The task's worktree. Made from the objective's current head on first claim, so it
        includes everything merged so far; a revision reuses it as is."""
        d = self.task_dir(task_id)
        if d.exists():
            return d
        obj = self.start_objective(root_id)
        base = self.head(obj)
        branch = self.task_branch(task_id)
        git(self.repo, "worktree", "add", "-q", "-b", branch, str(d), base)
        git(self.repo, "config", f"branch.{branch}.orchbase", base)
        return d

    def base(self, task_id: str) -> str:
        branch = self.task_branch(task_id)
        return git(self.repo, "config", f"branch.{branch}.orchbase").stdout.strip()

    def commit(self, task_id: str, message: str, *, author: str) -> str | None:
        """Commit everything in the task's worktree; None if nothing changed."""
        d = self.task_dir(task_id)
        git(d, "add", "-A")
        if git(d, "diff", "--cached", "--quiet", check=False).returncode == 0:
            return None
        git(d, "commit", "-q", "-m", message, f"--author={author} <{author}@orch>")
        return self.head(d)

    def changed_files(self, task_id: str) -> list[str]:
        """Files the task changed since its base, committed or not (new files included)."""
        d = self.task_dir(task_id)
        tracked = git(d, "diff", "--name-only", self.base(task_id)).stdout.splitlines()
        new = git(d, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
        return sorted({*tracked, *new})

    def revert_files(self, task_id: str, files: list[str]) -> None:
        """Put these files back as they were at the task's base: restored, or deleted if new."""
        d, base = self.task_dir(task_id), self.base(task_id)
        for f in files:
            if git(d, "cat-file", "-e", f"{base}:{f}", check=False).returncode == 0:
                git(d, "checkout", base, "--", f)
            else:
                (d / f).unlink(missing_ok=True)

    def diff(self, task_id: str, *, limit: int = 40_000) -> str:
        """The task's change since its base, for reviewers."""
        out = git(self.task_dir(task_id), "diff", self.base(task_id), "HEAD").stdout
        return out if len(out) <= limit else out[:limit] + f"\n... diff cut at {limit} chars"

    def is_merged(self, root_id: str, task_id: str) -> bool:
        branch = self.task_branch(task_id)
        if not self._has_branch(branch):
            return False
        obj = self.objective_branch(root_id)
        return (
            git(self.repo, "merge-base", "--is-ancestor", branch, obj, check=False).returncode == 0
        )

    def merge_task(self, root_id: str, task_id: str) -> Merge:
        """Merge a finished task into its objective branch; abort cleanly on a conflict."""
        obj = self.start_objective(root_id)
        branch = self.task_branch(task_id)
        p = git(obj, "merge", "--no-ff", "-q", "-m", f"Merge {branch}", branch, check=False)
        if p.returncode != 0:
            git(obj, "merge", "--abort", check=False)
            return Merge(False, (p.stdout + p.stderr).strip())
        return Merge(True, self.head(obj))


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}}
