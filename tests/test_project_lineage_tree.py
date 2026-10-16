from types import SimpleNamespace

import pytest

from ui.file_tree import artifact_step, build_project_model


def art(aid, parent, *, version=2):
    return SimpleNamespace(artifact_id=aid, line_id='L01', name=aid,
                           method_id='dewow', method_name='Dewow', created_at=aid,
                           manifest={'params': {'lineage_version': version,
                                     'step_input_artifact_id': parent,
                                     'run_group_id': 'run', 'run_step_index': 1}})


def test_saved_results_live_beneath_their_direct_input():
    line = SimpleNamespace(line_id='L01', name='L01')
    nodes = build_project_model([line], [art('A', ''), art('B', 'A'), art('C', 'A')])
    line_node = nodes[0].children[0]
    assert line_node.children, 'each line must expose raw input and its data descendants'
    raw = line_node.children[0]
    assert raw.kind == 'raw'
    assert [n.payload for n in raw.children] == ['A']
    assert [n.payload for n in raw.children[0].children] == ['B', 'C']


def test_unknown_and_cyclic_legacy_edges_are_not_fabricated():
    line = SimpleNamespace(line_id='L01', name='L01')
    nodes = build_project_model([line], [art('old', '', version=1), art('A', 'B'), art('B', 'A')])
    line_node = nodes[0].children[0]
    assert line_node.children, 'lineage and unknown origins need distinct containers'
    assert not line_node.children[0].children
    unknown = line_node.children[1]
    assert '来源待核对' in unknown.text
    assert {n.payload for n in unknown.children} == {'old', 'A', 'B'}


@pytest.mark.parametrize('lineage', [7, 'invalid', {'0': {}}, [None]])
def test_malformed_step_metadata_is_read_only(lineage):
    artifact = art('bad', '')
    artifact.manifest['params']['lineage'] = lineage
    assert artifact_step(artifact) == {}
