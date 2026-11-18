"""Phase 0 tests.

Offline (run by default):   pytest
Needs services running:     make up && pytest -m infra
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import download_data  # noqa: E402


# ---------- offline ----------
def test_compose_defines_required_services():
    doc = yaml.safe_load((ROOT / "infra" / "docker-compose.yml").read_text())
    for svc in ("postgres", "redis", "opensearch", "mediamtx"):
        assert svc in doc["services"], f"missing service {svc}"


@pytest.mark.parametrize("svc", ["postgres", "redis", "opensearch"])
def test_stateful_services_have_healthchecks(svc):
    doc = yaml.safe_load((ROOT / "infra" / "docker-compose.yml").read_text())
    assert "healthcheck" in doc["services"][svc]


def test_postgres_init_enables_pgvector():
    assert "CREATE EXTENSION IF NOT EXISTS vector" in (ROOT / "infra/postgres/init.sql").read_text()


def test_env_example_has_every_variable_compose_needs():
    text = (ROOT / ".env.example").read_text()
    for var in ("POSTGRES_PASSWORD", "POSTGRES_USER", "POSTGRES_DB"):
        assert var in text


def test_dataset_registry_is_well_formed():
    for name, d in download_data.DATASETS.items():
        assert d["method"] in ("url", "manual"), name
        assert d["expect"], name
        if d["method"] == "url":
            assert d["url"].startswith("http"), name
        else:
            assert d.get("how"), f"{name} needs manual instructions"


def test_starter_datasets_from_the_plan_are_registered():
    for name in ("mot17", "market1501", "cuhk03", "lfw", "rwf2000", "urfall", "weapons", "wildtrack"):
        assert name in download_data.DATASETS


def test_verify_detects_missing_and_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "RAW", tmp_path)
    assert download_data.is_present("lfw")[0] is False
    (tmp_path / "lfw" / "sub" / "lfw").mkdir(parents=True)
    assert download_data.is_present("lfw")[0] is True


def test_safe_extract_blocks_path_traversal(tmp_path):
    import zipfile
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError):
        download_data.safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_extracts_normal_zip(tmp_path):
    import zipfile
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("a/b.txt", "hello")
    download_data.safe_extract(good, tmp_path / "out")
    assert (tmp_path / "out" / "a" / "b.txt").read_text() == "hello"


# ---------- need docker compose up ----------
@pytest.mark.infra
def test_all_services_pass_check_script():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_infra.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.infra
def test_fake_camera_stream_is_readable():
    """Needs `make up-cams` and a data/sample_videos/sample.mp4."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_infra.py"), "--stream", "cam1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
