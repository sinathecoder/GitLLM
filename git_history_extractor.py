#!/usr/bin/env python3
"""
Git History Extractor
---------------------
Extracts the complete commit history — including the actual code changes (diffs)
for every commit — from a local Git repository and saves it to a text file.

Usage:
    python git_history_extractor.py /path/to/repo [--output history.txt] [--no-diff]
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config


def run_git(repo_path: Path, args: list[str]) -> str:
    """Run a git command in the given repository and return its stdout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path)] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def get_commit_count(repo_path: Path) -> int:
    """Return the total number of commits across all branches in the repository."""
    output = run_git(repo_path, ["rev-list", "--count", "--all"])
    return int(output.strip())


def get_history_with_diffs(repo_path: Path) -> str:
    """
    Get the full commit history including the complete code changes (diffs)
    for every commit. This uses `git log -p` which walks the full history
    and includes the exact added/removed lines for each file in each commit.
    """
    return run_git(
        repo_path,
        [
            "log",
            "--all",
            "--date=iso",
            "--patch",
            "--pretty=format:"
            "============================================================\n"
            "Commit: %H\n"
            "Abbreviated: %h\n"
            "Author: %an <%ae>\n"
            "Date: %ad\n"
            "Subject: %s\n"
            "Body:\n"
            "%b"
            "============================================================",
        ],
    )


def get_history_metadata_only(repo_path: Path) -> str:
    """Get the full commit history with detailed metadata but no diffs."""
    return run_git(
        repo_path,
        [
            "log",
            "--all",
            "--date=iso",
            "--pretty=format:"
            "Commit: %H%n"
            "Abbreviated: %h%n"
            "Author: %an <%ae>%n"
            "Date: %ad%n"
            "Subject: %s%n"
            "Body:%n%b"
            "----------------------------------------",
        ],
    )


def get_branches(repo_path: Path) -> str:
    """Get the list of all branches."""
    return run_git(repo_path, ["branch", "-a"])


def get_remotes(repo_path: Path) -> str:
    """Get the list of remote repositories."""
    return run_git(repo_path, ["remote", "-v"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract git history (including all code changes) from a local repository and save to a text file.",
        epilog="Example: python git_history_extractor.py /path/to/repo -o history.txt",
    )
    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to the local Git repository folder",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=config.HISTORY_OUTPUT_DEFAULT,
        help=f"Output text file path (default: {config.HISTORY_OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Only include commit metadata (subject, author, date), not the actual code changes",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()

    # Validate the repository path
    if not repo_path.exists():
        print(f"Error: Path does not exist: {repo_path}", file=sys.stderr)
        sys.exit(1)

    if not (repo_path / ".git").exists():
        print(f"Error: Not a Git repository: {repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing repository: {repo_path}")

    # Gather repository metadata
    commit_count = get_commit_count(repo_path)
    branches = get_branches(repo_path)
    remotes = get_remotes(repo_path)

    # Select the history output (with diffs by default)
    if args.no_diff:
        history = get_history_metadata_only(repo_path)
        history_label = "COMMIT HISTORY (metadata only, no diffs)"
        print("Mode: metadata only (--no-diff)")
    else:
        history = get_history_with_diffs(repo_path)
        history_label = "COMMIT HISTORY WITH ALL CODE CHANGES (diffs)"
        print("Mode: full history with code changes (diffs)")

    # Build the output content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_content = f"""========================================
GIT HISTORY REPORT — WITH ALL CHANGES
========================================
Repository: {repo_path}
Generated:  {timestamp}
Total commits: {commit_count}

----------------------------------------
REMOTES
----------------------------------------
{remotes}

----------------------------------------
BRANCHES
----------------------------------------
{branches}

----------------------------------------
{history_label}
----------------------------------------
{history}
"""

    # Write to file
    output_path = Path(args.output).resolve()
    output_path.write_text(output_content, encoding="utf-8")

    file_size = output_path.stat().st_size
    print(f"Successfully saved {commit_count} commits to: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"Total lines: {sum(1 for _ in output_path.open(encoding='utf-8')):,}")


if __name__ == "__main__":
    main()