"""Resume must account for every failed job before reporting success."""
import json
from types import SimpleNamespace

import pytest

import cli_batch


@pytest.mark.parametrize("summary", [{}, [], {"results": {}}, {"results": [None]},
                                     {"results": [{"job_id": "A"}]},
                                     {"results": [{"status": "failed"}]}])
def test_malformed_summary_is_rejected_without_running_jobs(tmp_path, monkeypatch, summary):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    ran = []
    monkeypatch.setattr(cli_batch, "run_job", lambda *args, **kwargs: ran.append(args))
    assert cli_batch.cmd_resume(SimpleNamespace(summary=str(path), repo_root=str(tmp_path))) == 2
    assert ran == []


def test_missing_failed_job_in_config_rejects_the_whole_resume(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"jobs": [{"id": "A", "input": "a.csv"}]}), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"config": str(config), "results": [
        {"job_id": "A", "status": "failed"}, {"job_id": "B", "status": "failed"},
    ]}), encoding="utf-8")
    ran = []
    monkeypatch.setattr(cli_batch, "run_job", lambda *args, **kwargs: ran.append(args) or {"status": "ok"})
    assert cli_batch.cmd_resume(SimpleNamespace(summary=str(summary), repo_root=str(tmp_path))) == 2
    assert ran == []
    assert "B" in capsys.readouterr().out


@pytest.mark.parametrize("results", [[], [{"job_id": "A", "status": "ok"}],
                                   [{"job_id": "A", "status": "success"}]])
def test_valid_legacy_summary_with_no_failures_is_successful_noop(tmp_path, results):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    assert cli_batch.cmd_resume(SimpleNamespace(summary=str(path), repo_root=str(tmp_path))) == 0


@pytest.mark.parametrize("stats", [{"ok": 0, "failed": 1, "total": 1},
                                   {"ok": 0, "failed": 0, "total": 1},
                                   {"ok": 0, "failed": 0},
                                   {"ok": 0, "failed": False, "total": 0}])
def test_present_stats_must_agree_with_summary_results(tmp_path, stats):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"results": [], "stats": stats}), encoding="utf-8")
    assert cli_batch.cmd_resume(SimpleNamespace(summary=str(path), repo_root=str(tmp_path))) == 2


def test_complete_resume_runs_all_failures_and_records_actual_results(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"jobs": [{"id": "A", "input": "a.csv"},
                                           {"id": "B", "input": "b.csv"}]}), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"config": str(config), "output_dir": str(tmp_path / "out"), "results": [
        {"job_id": "A", "status": "failed"}, {"job_id": "B", "status": "failed"},
    ]}), encoding="utf-8")
    ran = []

    def run(job, **_kwargs):
        ran.append(job["id"])
        return {"job_id": job["id"], "status": "ok"}

    monkeypatch.setattr(cli_batch, "run_job", run)
    assert cli_batch.cmd_resume(SimpleNamespace(summary=str(summary), repo_root=str(tmp_path))) == 0
    assert ran == ["A", "B"]
    written = json.loads(next((tmp_path / "out").glob("summary_*.json")).read_text(encoding="utf-8"))
    assert written["stats"] == {"ok": 2, "failed": 0, "total": 2}
    assert {row["job_id"] for row in written["results"]} == {"A", "B"}
