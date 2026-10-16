"""Source hygiene must distinguish local output from release source."""
from scripts.check_release_hygiene import scan


def test_local_build_output_is_not_scanned(tmp_path):
    cache = tmp_path / 'output' / 'installer' / 'localpycs'
    cache.mkdir(parents=True)
    (cache / 'module.pyc').write_bytes(b'local build cache')
    assert scan(tmp_path)['ok']


def test_source_and_top_level_build_debris_remain_forbidden(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core' / 'module.pyc').write_bytes(b'cache')
    (tmp_path / 'build').mkdir()
    assert scan(tmp_path)['findings'] == ['build', 'core/module.pyc']


def test_only_root_output_is_excluded(tmp_path):
    nested = tmp_path / 'core' / 'output'
    nested.mkdir(parents=True)
    (nested / 'module.pyc').write_bytes(b'cache')
    assert scan(tmp_path)['findings'] == ['core/output/module.pyc']
