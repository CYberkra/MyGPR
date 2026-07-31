import json
import logging
import zipfile
from pathlib import Path

from core.observability import build_support_bundle, configure_structured_logging, diagnostic_context


def test_structured_log_contains_context(tmp_path: Path):
    path = configure_structured_logging(tmp_path)
    logger = logging.getLogger("mygpr.test")
    with diagnostic_context(project_id="P1", line_id="L1"):
        logger.warning("hello")
    for handler in logging.getLogger("mygpr").handlers:
        handler.flush()
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["project_id"] == "P1"
    assert row["line_id"] == "L1"


def test_support_bundle_excludes_raw_data(tmp_path: Path):
    root = tmp_path / "project"
    (root / "logs").mkdir(parents=True)
    (root / "raw").mkdir()
    (root / "logs" / "app.log").write_text("ok", encoding="utf-8")
    (root / "raw" / "large.npy").write_bytes(b"secret")
    bundle = build_support_bundle(root, tmp_path / "support.zip")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    assert "logs/app.log" in names
    assert all(not name.startswith("raw/") for name in names)
