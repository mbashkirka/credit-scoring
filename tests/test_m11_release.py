from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_m11_portfolio_docs_exist():
    expected = [
        "README.md",
        "CHANGELOG.md",
        "docs/ARCHITECTURE.md",
        "docs/MODEL_CARD.md",
        "docs/RESULTS.md",
        "docs/MONITORING_CASE_STUDY.md",
        "docs/GOVERNANCE.md",
        "docs/REPRODUCIBILITY.md",
        "docs/DEMO.md",
        "scripts/validate_m11_release.py",
    ]
    assert all((ROOT / path).is_file() for path in expected)


def test_readme_reports_final_lockbox_and_monitoring_results():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "0.7168" in text
    assert "0.000739" in text
    assert "0.73532 → 0.71710" in text
    assert "OPENED_AND_EVALUATED" in text


def test_model_card_preserves_key_limitations():
    text = (ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8").lower()
    assert "not intended for real lending decisions" in text
    assert "not causal" in text
    assert "no verified timestamps" in text
    assert "illustrative" in text


def test_monitoring_case_study_is_explicitly_retrospective():
    text = (ROOT / "docs" / "MONITORING_CASE_STUDY.md").read_text(encoding="utf-8").lower()
    assert "retrospective replay" in text
    assert "does not" in text and "caused" in text
    assert "2 warnings" in text or "2 warning" in text


def test_release_validator_is_wired_into_makefile_and_ci():
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "release-check:" in make
    assert "validate_m11_release.py" in make
    assert "M11 source release contract" in ci


def test_m11_release_manifest_freezes_model_identity():
    import json

    data = json.loads((ROOT / "release" / "m11_release_manifest.json").read_text(encoding="utf-8"))
    assert data["model_version"] == "credit-risk-v1"
    assert data["model_artifacts_changed"] is False
    assert data["lockbox_status"] == "OPENED_AND_EVALUATED"
    assert data["monitoring_case_study"] == "retrospective"


def test_gitignore_protects_all_local_data_and_artifacts():
    rules = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        "data/**",
        "!data/**/",
        "!data/**/.gitkeep",
        "artifacts/**",
        "!artifacts/.gitkeep",
    }.issubset(rules)


def test_release_check_allows_local_untracked_ml_state():
    import subprocess
    import sys

    artifact = ROOT / "artifacts" / "__m11_release_test__.joblib"
    processed = ROOT / "data" / "processed" / "__m11_release_test__.parquet"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"local model fixture")
    processed.write_bytes(b"local processed-data fixture")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/validate_m11_release.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "M11 RELEASE CONTRACT: PASS" in result.stdout
    finally:
        artifact.unlink(missing_ok=True)
        processed.unlink(missing_ok=True)
