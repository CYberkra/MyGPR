"""Persisted processing lineage: saved edges, independent runs and legacy reads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.processing_artifact_index import ProcessingArtifactRecord
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.infrastructure.persistence.artifact_delete import collect_artifact_descendants
from mygpr.interfaces.backend import MyGPRBackend


@pytest.fixture
def project(tmp_path: Path):
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / "lineage", name="Lineage")
    data = np.random.default_rng(41).normal(size=(72, 48)).astype(np.float32)
    backend.projects.save_line_dataset(summary.project_id, "L01", data, time_window_ns=500.0)
    try:
        yield backend, summary, data
    finally:
        backend.shutdown()


def _pipeline(window: int = 11) -> PipelineDefinition:
    return PipelineDefinition(name="two steps", steps=(
        PipelineStep("dewow", {"window": window}),
        PipelineStep("agcGain", {"window": 9}),
    ))


def _params(summary, artifact) -> dict:
    path = Path(summary.root_path) / artifact.manifest["params_path"]
    return json.loads(path.read_text(encoding="utf-8"))["params"]


def _run_artifacts(backend, summary, final):
    run_id = _params(summary, final)["run_group_id"]
    return sorted((item for item in backend.projects.list_artifacts(summary.project_id, "L01")
                   if _params(summary, item).get("run_group_id") == run_id),
                  key=lambda item: _params(summary, item)["run_step_index"])


def test_continue_from_saved_intermediate_records_direct_edges_and_preserves_old_run(project):
    backend, summary, _ = project
    pid = summary.project_id
    old_final = backend.project_processing.execute_pipeline(pid, "L01", _pipeline())
    old_steps = _run_artifacts(backend, summary, old_final)
    source = old_steps[0]
    source_data = backend.projects.read_artifact_dataset(pid, "L01", source.artifact_id)
    old_hashes = {item.artifact_id: item.sha256 for item in old_steps}
    expected = backend.processing.execute_pipeline(
        source_data.data, _pipeline(15), header_info=source_data.header_info,
        trace_metadata=source_data.trace_metadata,
    )

    final = backend.project_processing.execute_pipeline(
        pid, "L01", _pipeline(15), input_artifact_id=source.artifact_id,
    )
    steps = _run_artifacts(backend, summary, final)
    assert len(steps) == 2  # final is saved once, alongside one intermediate
    assert final.branch_id != old_final.branch_id
    assert len({item.branch_id for item in steps}) == 1
    for index, artifact in enumerate(steps, start=1):
        params = _params(summary, artifact)
        assert params["lineage_version"] == 2
        assert params["run_input_artifact_id"] == source.artifact_id
        assert params["step_input_artifact_id"] == (source if index == 1 else steps[0]).artifact_id
        assert params["run_step_index"] == index
        assert params["run_input_sha256"] == source.sha256
        assert params["step_input_sha256"] == (source if index == 1 else steps[0]).sha256
        assert artifact.parent_artifact_id == source.artifact_id  # legacy run-input contract
        assert artifact.manifest["params"] == params
        assert params["lineage"][-1]["implementation_version"]
        assert params["lineage"][-1]["params"]
    loaded = backend.projects.read_artifact_dataset(pid, "L01", final.artifact_id)
    np.testing.assert_array_equal(loaded.data, expected.data)
    by_id = {item.artifact_id: item for item in backend.projects.list_artifacts(pid, "L01")}
    assert {aid: by_id[aid].sha256 for aid in old_hashes} == old_hashes


def test_deleting_one_saved_branch_keeps_sibling_and_its_data(project):
    backend, summary, _ = project
    pid = summary.project_id
    source = backend.project_processing.execute_pipeline(
        pid, "L01", PipelineDefinition(steps=(PipelineStep("dewow", {"window": 7}),)),
    )
    final = backend.project_processing.execute_pipeline(
        pid, "L01", _pipeline(), input_artifact_id=source.artifact_id,
    )
    first, _ = _run_artifacts(backend, summary, final)
    sibling = backend.project_processing.execute_pipeline(
        pid, "L01", _pipeline(19), input_artifact_id=source.artifact_id,
    )
    siblings = _run_artifacts(backend, summary, sibling)
    before = backend.projects.read_artifact_dataset(pid, "L01", sibling.artifact_id).data.copy()
    assert set(backend.list_artifact_descendants(pid, "L01", first.artifact_id)) == {
        first.artifact_id, final.artifact_id,
    }
    deleted = backend.delete_artifacts(pid, "L01", [first.artifact_id])
    assert set(deleted.deleted_artifact_ids) == {first.artifact_id, final.artifact_id}
    remaining = {item.artifact_id for item in backend.projects.list_artifacts(pid, "L01")}
    assert remaining == {source.artifact_id, *(item.artifact_id for item in siblings)}
    assert set(backend.list_artifact_descendants(pid, "L01", source.artifact_id)) == remaining
    np.testing.assert_array_equal(
        backend.projects.read_artifact_dataset(pid, "L01", sibling.artifact_id).data, before,
    )
    backend.delete_artifacts(pid, "L01", [source.artifact_id])
    assert backend.projects.list_artifacts(pid, "L01") == ()
    assert all(not branch["head_artifact_id"] for branch in
               backend.projects._session(pid)._store.list_processing_branches("L01"))


def test_block_pipeline_saves_only_actual_final_node(project):
    backend, summary, _ = project
    final = backend.project_processing.execute_pipeline(
        summary.project_id, "L01", _pipeline(), save_intermediates=False,
    )
    params = _params(summary, final)
    assert backend.projects.list_artifacts(summary.project_id, "L01") == (final,)
    assert params["execution_mode"] == "file_backed_blocks"
    assert params["lineage_version"] == 2
    assert params["run_step_index"] == 2
    assert params["run_input_artifact_id"] == params["step_input_artifact_id"] == ""
    assert params["run_input_sha256"] == params["step_input_sha256"]
    assert params["run_input_sha256"]
    assert len(params["lineage"]) == 2  # records do not become selectable artifacts
    assert all("artifact_id" not in step for step in params["lineage"])


def test_loaded_fallback_honors_final_only_save_policy(project, monkeypatch):
    backend, summary, _ = project
    monkeypatch.setattr(backend.processing, "supports_block_pipeline", lambda pipeline: False)
    final = backend.project_processing.execute_pipeline(
        summary.project_id, "L01", _pipeline(), save_intermediates=False,
    )
    assert backend.projects.list_artifacts(summary.project_id, "L01") == (final,)
    params = _params(summary, final)
    assert params["execution_mode"] == "loaded"
    assert params["run_step_index"] == 2
    assert params["step_input_artifact_id"] == ""


def test_explicit_branch_does_not_change_raw_run_input(project):
    backend, summary, _ = project
    first = backend.project_processing.execute_pipeline(
        summary.project_id, "L01", _pipeline(), branch_id="L01:chosen",
    )
    second = backend.project_processing.execute_pipeline(
        summary.project_id, "L01", _pipeline(13), branch_id="L01:chosen",
    )
    assert first.artifact_id != second.artifact_id
    assert first.branch_id == second.branch_id == "L01:chosen"
    for artifact in backend.projects.list_artifacts(summary.project_id, "L01"):
        assert artifact.parent_artifact_id == ""
        assert _params(summary, artifact)["run_input_artifact_id"] == ""


def test_legacy_dependency_fallback_does_not_invent_step_edges(project):
    backend, summary, data = project
    pid = summary.project_id
    source = backend.projects.save_processing_artifact(
        pid, "L01", data, name="legacy root", method_id="legacy", method_name="legacy",
        params={}, pipeline=[], branch_id="L01:legacy-root", input_dataset={},
    )
    old = backend.projects.save_processing_artifact(
        pid, "L01", data, name="legacy child", method_id="legacy", method_name="legacy",
        params={"parent_artifact_id": source.artifact_id, "run_step_index": 2},
        pipeline=[], branch_id="L01:legacy-child", input_dataset={},
    )
    assert "lineage_version" not in old.manifest.get("params", {})
    new = backend.project_processing.execute_pipeline(
        pid, "L01", _pipeline(), input_artifact_id=old.artifact_id,
    )
    assert set(backend.list_artifact_descendants(pid, "L01", source.artifact_id)) == {
        source.artifact_id, old.artifact_id, *(a.artifact_id for a in _run_artifacts(backend, summary, new)),
    }
    records = [
        ProcessingArtifactRecord("root", "L01", "root.npy"),
        ProcessingArtifactRecord("one", "L01", "one.npy", parent_artifact_id="root", run_group_id="run"),
        ProcessingArtifactRecord("two", "L01", "two.npy", parent_artifact_id="root", run_group_id="run"),
    ]
    assert collect_artifact_descendants(records, "one") == ("one",)


def test_delete_final_restores_branch_head_to_saved_predecessor(project):
    backend, summary, _ = project
    pid = summary.project_id
    final = backend.project_processing.execute_pipeline(pid, "L01", _pipeline())
    first, _ = _run_artifacts(backend, summary, final)
    backend.delete_artifacts(pid, "L01", [final.artifact_id])
    branches = backend.projects._session(pid)._store.list_processing_branches("L01")
    assert next(branch for branch in branches if branch["branch_id"] == final.branch_id)[
        "head_artifact_id"
    ] == first.artifact_id


def test_intermediate_cleanup_retains_materialized_v2_dependencies(project):
    backend, summary, _ = project
    pid = summary.project_id
    final = backend.project_processing.execute_pipeline(pid, "L01", _pipeline())
    first, _ = _run_artifacts(backend, summary, final)
    assert backend.projects.clear_intermediate_artifacts(pid, "L01") == ()
    assert set(backend.list_artifact_descendants(pid, "L01", first.artifact_id)) == {
        first.artifact_id, final.artifact_id,
    }
    assert backend.projects.read_artifact_dataset(pid, "L01", first.artifact_id).data.shape == (72, 48)


@pytest.mark.parametrize('with_legacy_child', [False, True])
def test_cleanup_preserves_legacy_sources_and_branch_heads(project, with_legacy_child):
    backend, summary, data = project
    pid = summary.project_id
    source = backend.projects.save_processing_artifact(
        pid, 'L01', data, name='legacy intermediate', method_id='legacy', method_name='legacy',
        params={'artifact_kind': 'intermediate'}, pipeline=[], branch_id='legacy-source', input_dataset={},
    )
    if with_legacy_child:
        child = backend.projects.save_processing_artifact(
            pid, 'L01', data, name='legacy child', method_id='legacy', method_name='legacy',
            params={'parent_artifact_id': source.artifact_id}, pipeline=[],
            branch_id='legacy-child', input_dataset={},
        )
        # Exercise the legacy reference independently of branch-head protection.
        backend.projects._session(pid)._store.storage.catalog.ensure_branch(
            branch_id='legacy-source', line_id='L01', name='legacy-source', head_artifact_id=child.artifact_id)
    assert backend.projects.clear_intermediate_artifacts(pid, 'L01') == ()
    np.testing.assert_array_equal(
        backend.projects.read_artifact_dataset(pid, 'L01', source.artifact_id).data, data)
    catalog = backend.projects._session(pid)._store.storage.catalog
    assert catalog.get_artifact(catalog.branch_head('legacy-source')) is not None


def test_batch_delete_orders_overlapping_targets_by_dependency(project):
    backend, summary, _ = project
    pid = summary.project_id
    source = backend.project_processing.execute_pipeline(
        pid, "L01", PipelineDefinition(steps=(PipelineStep("dewow", {"window": 7}),)),
    )
    final = backend.project_processing.execute_pipeline(
        pid, "L01", _pipeline(), input_artifact_id=source.artifact_id,
    )
    result = backend.delete_artifacts(pid, "L01", [final.artifact_id, source.artifact_id])
    assert len(result.deleted_artifact_ids) == 3
    assert backend.projects.list_artifacts(pid, "L01") == ()
