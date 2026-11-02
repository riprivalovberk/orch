"""Git workspace for one target repo: a branch per objective, a worktree per work task.

    main ──► obj/<root>            the objective's integration branch (own worktree)
               ├─► task/<task id>  one per test/implement task, cut from obj/<root> when
               └─► task/<task id>  the task is first claimed; merged back once it is DONE

Agents in parallel never share a directory. PAR-001 keeps parallel tasks off the same
files, so merges are clean. Once the objective is approved, obj/<root> merges into main.
"""

from __future__ import annotations

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
    def __init__(self, repo: Path, worktrees: Path) -> None:
        self.repo = repo.resolve()
        self.worktrees = worktrees.resolve()

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
        committed = git(d, "diff", "--name-only", self.base(task_id), "HEAD").stdout.split()
        pending = git(d, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
        return sorted({*committed, *(line[3:].strip() for line in pending)})

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
