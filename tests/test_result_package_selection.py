from dataclasses import replace

import pytest

from mygpr.domain.project.models import ProjectArtifact
from mygpr.domain.project.result_package import select_saved_steps


def artifact(aid, parent='', **params):
    return ProjectArtifact(aid, 'L01', aid, '', manifest={'params': dict(
        lineage_version=2, step_input_artifact_id=parent, **params)})


def test_legacy_parent_is_not_used_as_a_confirmed_ancestry_edge():
    first = artifact('first')
    old = replace(artifact('old'), parent_artifact_id='first', manifest={'params': {'parent_artifact_id': 'first'}})
    selected, warnings = select_saved_steps([first, old], 'old')
    assert [a.artifact_id for a in selected] == ['old']
    assert warnings and '缺少可靠' in warnings[0]


def test_missing_parent_is_recorded_without_guessing_another_step():
    selected, warnings = select_saved_steps([artifact('child', 'removed'), artifact('sibling')], 'child')
    assert [a.artifact_id for a in selected] == ['child']
    assert 'removed' in warnings[0]


@pytest.mark.parametrize('target', ['', 'a'])
def test_cycle_cannot_be_published_as_a_valid_chain(target):
    with pytest.raises(ValueError, match='循环'):
        select_saved_steps([artifact('a', 'b'), artifact('b', 'a')], target)


def test_deep_chain_is_iterative():
    selected, warnings = select_saved_steps([artifact(str(i), str(i-1) if i else '') for i in range(1100)], '1099')
    assert len(selected) == 1100 and not warnings
    assert selected[0].artifact_id == '0' and selected[-1].artifact_id == '1099'


def test_package_h5_schema_is_distinct_from_legacy_json_and_has_one_owner():
    import json
    from pathlib import Path
    from core.project_models import ProcessingResultV1
    from mygpr.domain.project.result_package import RESULT_H5_SCHEMA
    assert ProcessingResultV1.__dataclass_fields__['schema'].default != RESULT_H5_SCHEMA
    payload = json.loads((Path(__file__).resolve().parents[1] / 'config/schema_catalog.json').read_text(encoding='utf-8'))
    assert sum(row['schema'] == RESULT_H5_SCHEMA for row in payload['schemas']) == 1
