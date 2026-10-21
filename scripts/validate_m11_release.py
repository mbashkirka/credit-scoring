#!/usr/bin/env python3
"""Validate the M11 Git/public source-release contract.

Local datasets and model/evaluation artifacts are expected to exist in a working ML
repository. They are not a release violation by themselves. The public-release
contract instead requires that these paths are ignored by Git and, when the project
is already a Git worktree, that no prohibited payload is tracked.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    "README.md",
    "CHANGELOG.md",
    "docs/ARCHITECTURE.md",
    "docs/MODEL_CARD.md",
    "docs/RESULTS.md",
    "docs/MONITORING_CASE_STUDY.md",
    "docs/GOVERNANCE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/DEMO.md",
    "release/m11_release_manifest.json",
]

REQUIRED_README_FACTS = [
    "0.7168",
    "2.84e-09",
    "OPENED_AND_EVALUATED",
    "0.000739",
    "not a production lending system",
]

FORBIDDEN_SOURCE_SUFFIXES = {".cbm", ".joblib", ".pq", ".parquet"}
LOCAL_ONLY_ROOTS = {"artifacts", "data"}
ALLOWED_LOCAL_PLACEHOLDER = ".gitkeep"
REQUIRED_GITIGNORE_RULES = {
    "data/**",
    "!data/**/",
    "!data/**/.gitkeep",
    "artifacts/**",
    "!artifacts/.gitkeep",
}


def fail(message: str) -> None:
    raise SystemExit(f"M11 RELEASE CHECK: FAIL — {message}")


def check_required_files() -> None:
    missing = [p for p in REQUIRED_DOCS if not (ROOT / p).is_file()]
    if missing:
        fail(f"missing required documentation: {missing}")


def check_version() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = data["project"]["version"]
    if version != "0.11.0":
        fail(f"expected package version 0.11.0, found {version}")


def check_release_manifest() -> None:
    data = json.loads((ROOT / "release" / "m11_release_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "release_stage": "M11",
        "package_version": "0.11.0",
        "model_version": "credit-risk-v1",
        "model_artifact_fingerprint": "14a17e7ac80680624a0587f9cf009b0d4e0f61b98eec300fe5297087c496f301",
        "model_artifacts_changed": False,
        "lockbox_status": "OPENED_AND_EVALUATED",
        "monitoring_case_study": "retrospective",
        "source_only_release": True,
        "contains_raw_data": False,
        "contains_model_binaries": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            fail(f"release manifest mismatch for {key}: expected {value!r}, found {data.get(key)!r}")


def check_readme_facts() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    for fact in REQUIRED_README_FACTS:
        if fact.lower() not in lowered:
            fail(f"README missing release fact/guardrail: {fact!r}")


def check_markdown_links() -> None:
    docs = [ROOT / p for p in REQUIRED_DOCS]
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    broken: list[str] = []
    for md in docs:
        text = md.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            resolved = (md.parent / clean).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    if broken:
        fail(f"broken internal Markdown links: {broken}")


def _is_allowed_placeholder(rel: Path) -> bool:
    return rel.name == ALLOWED_LOCAL_PLACEHOLDER and rel.parts and rel.parts[0] in LOCAL_ONLY_ROOTS


def _is_prohibited_public_path(rel: Path) -> bool:
    """Return True when a path must never be part of the public source release."""
    if _is_allowed_placeholder(rel):
        return False
    if rel.parts and rel.parts[0] in LOCAL_ONLY_ROOTS:
        return True
    return rel.suffix.lower() in FORBIDDEN_SOURCE_SUFFIXES


def _git_tracked_files() -> list[Path] | None:
    """Return tracked paths, or None when ROOT is not an initialized Git worktree."""
    try:
        probe = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return None

    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(p.decode("utf-8")) for p in result.stdout.split(b"\0") if p]


def check_gitignore_policy() -> None:
    path = ROOT / ".gitignore"
    if not path.is_file():
        fail("missing .gitignore")
    rules = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(REQUIRED_GITIGNORE_RULES - rules)
    if missing:
        fail(f".gitignore is missing local-data/artifact protection rules: {missing}")


def check_source_only_release() -> tuple[str, int]:
    """Validate what can be published, without rejecting legitimate local ML state.

    Local files below data/ and artifacts/ are deliberately allowed to remain on disk.
    If Git is initialized, the authoritative publication surface is the tracked file set.
    Without Git, we enforce the ignore policy and scan every non-local-only path for
    prohibited serialized model/data suffixes.
    """
    tracked = _git_tracked_files()
    if tracked is not None:
        offenders = sorted(str(rel) for rel in tracked if _is_prohibited_public_path(rel))
        if offenders:
            fail(
                "Git tracks data/model artifacts that must stay local: "
                f"{offenders}. Remove them from the index (for example with "
                "`git rm -r --cached data artifacts`) while keeping the local files."
            )
        return "git-tracked-files", len(tracked)

    offenders: list[str] = []
    checked = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        # data/ and artifacts/ are local-only working-state roots and are governed by
        # .gitignore. Their physical presence is expected and is not a source-release error.
        if rel.parts and rel.parts[0] in LOCAL_ONLY_ROOTS:
            continue
        checked += 1
        if rel.suffix.lower() in FORBIDDEN_SOURCE_SUFFIXES:
            offenders.append(str(rel))
    if offenders:
        fail(f"public source tree contains model/data payload outside local-only roots: {offenders}")
    return "filesystem-fallback", checked


def check_governance_language() -> None:
    corpus = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in REQUIRED_DOCS)
    required = [
        "chunks 10–11 are now historical evaluation data",
        "does not claim temporal drift",
        "not causal",
        "retrospective",
    ]
    for phrase in required:
        if phrase.lower() not in corpus.lower():
            fail(f"missing governance wording: {phrase!r}")


def main() -> int:
    check_required_files()
    check_version()
    check_release_manifest()
    check_readme_facts()
    check_markdown_links()
    check_gitignore_policy()
    source_mode, source_count = check_source_only_release()
    check_governance_language()
    print("=== M11 RELEASE CONTRACT ===")
    print("required docs: PASS")
    print("internal Markdown links: PASS")
    print("version 0.11.0: PASS")
    print("release manifest: PASS")
    print("Model V1 / lockbox / monitoring guardrails: PASS")
    print(".gitignore local-data/artifact policy: PASS")
    print(f"publication surface: PASS ({source_mode}, checked={source_count})")
    print("source-only packaging: PASS")
    print("M11 RELEASE CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
