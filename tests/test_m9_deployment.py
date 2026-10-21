from pathlib import Path

from credit_risk.api.runtime import ArtifactPaths


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARTIFACTS = {
    "m4_catboost.cbm",
    "m4_catboost_imputer.joblib",
    "m4_calibrator.joblib",
    "m3_features.csv",
    "m4_metadata.json",
    "m5_policy_freeze.json",
    "m5_risk_band_thresholds.json",
}


def test_artifact_paths_honors_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CREDIT_RISK_ARTIFACT_DIR", str(tmp_path))
    paths = ArtifactPaths()
    assert paths.root == tmp_path
    assert {path.name for path in paths.frozen_inputs()} == EXPECTED_ARTIFACTS


def test_serving_artifact_manifest_matches_runtime_contract():
    manifest = ROOT / "deployment" / "serving-artifacts.txt"
    names = {
        line.strip().split("/", 1)[-1]
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert names == EXPECTED_ARTIFACTS


def test_dockerfile_is_non_root_and_has_readiness_healthcheck():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER 10001:10001" in text
    assert "HEALTHCHECK" in text
    assert "127.0.0.1:8000/ready" in text
    assert "COPY data" not in text


def test_dockerfile_copies_exact_frozen_serving_artifacts():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    copied = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("COPY artifacts/"):
            source = line.split()[1]
            copied.add(Path(source).name)
    assert copied == EXPECTED_ARTIFACTS


def test_dockerignore_excludes_data_and_whitelists_only_serving_artifacts():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "data" in lines
    allowed = {
        Path(line[1:]).name
        for line in lines
        if line.startswith("!artifacts/")
    }
    assert allowed == EXPECTED_ARTIFACTS


def test_pyproject_uses_httpx2_for_testclient_stack():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"httpx2>=2.10,<3"' in text
    assert '"httpx>=' not in text


def test_ci_has_real_container_contract_job():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "generate_m9_ci_artifacts.py" in text
    assert "docker build" in text
    assert "m9_container_smoke_test.py" in text
    assert 'id -u' in text


def test_compose_hardens_serving_container_and_uses_readiness_healthcheck():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "read_only: true" in text
    assert "cap_drop:" in text and "- ALL" in text
    assert "no-new-privileges:true" in text
    assert "127.0.0.1:8000/ready" in text


def test_runtime_requirements_are_direct_pins_and_exclude_research_only_packages():
    lines = {
        line.strip()
        for line in (ROOT / "requirements" / "runtime.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "catboost==1.2.8" in lines
    assert "scikit-learn==1.9.0" in lines
    assert "fastapi==0.141.1" in lines
    assert "starlette==1.6.0" in lines
    assert not any(line.startswith("pytest") for line in lines)
    assert not any(line.startswith("kaggle") for line in lines)
    assert not any(line.startswith("matplotlib") for line in lines)


def test_dev_requirements_pin_httpx2_test_stack():
    text = (ROOT / "requirements" / "dev.txt").read_text(encoding="utf-8")
    assert "httpx2==2.10.0" in text
    assert "httpx==" not in text


def test_docker_python_base_is_patch_pinned():
    first = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()[1]
    assert first == "FROM python:3.12.11-slim-bookworm"


def test_runtime_rejects_inconsistent_sklearn_pickle_versions():
    text = (ROOT / "src" / "credit_risk" / "api" / "runtime.py").read_text(encoding="utf-8")
    assert "InconsistentVersionWarning" in text
    assert 'warnings.simplefilter("error", InconsistentVersionWarning)' in text
