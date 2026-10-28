"""Release assets must retain application branding and valid acquisition metadata."""
import importlib.util
import importlib.metadata
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest


def recipe(name):
    path = Path(__file__).parents[1] / 'packaging' / 'windows' / (name + '.py')
    spec = importlib.util.spec_from_file_location('windows_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_executable_icon_uses_the_source_application_logo(tmp_path):
    source = tmp_path / 'source'
    resource = source / 'ui/resources/mygpr_logo.png'
    resource.parent.mkdir(parents=True)
    Image.new('RGBA', (256, 256), (14, 90, 130, 255)).save(resource)
    before = resource.read_bytes()
    output = tmp_path / 'app.ico'
    recipe('build').create_app_icon(source, output)
    with Image.open(output) as icon:
        assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= icon.ico.sizes()
        assert icon.convert('RGBA').getpixel((128, 128)) == (14, 90, 130, 255)
    assert resource.read_bytes() == before


@pytest.mark.parametrize('endpoint', [False, True])
@pytest.mark.parametrize('height_present', [False, True])
def test_package_check_accepts_valid_axes_and_keeps_trace_identity(tmp_path, monkeypatch, endpoint, height_present):
    from mygpr.interfaces.cli import backend_smoke
    import json

    monkeypatch.setattr(backend_smoke, 'run_smoke', lambda **kwargs: {'finite': True})
    version = (Path(__file__).parents[1] / 'VERSION').read_text().strip()
    monkeypatch.setattr(importlib.metadata, 'version', lambda _name: version)
    path = tmp_path / 'completed.h5'
    values = np.arange(3 * 501, dtype=np.float32).reshape(3, 501) - 700
    trace_ids = np.arange(2**53 + 1, 2**53 + 4, dtype=np.uint64)
    with h5py.File(path, 'w') as handle:
        handle.attrs.update(samples_per_trace=501, time_window_ns=900., store_component='real',
                            committed_trace_count=3, time_axis_endpoint=endpoint)
        handle.create_dataset('traces', data=values)
        handle.create_dataset('trace_meta/trace_id', data=trace_ids)
        if height_present:
            handle.create_dataset('trace_meta/height', data=[2., np.nan, np.inf])
    before = path.read_bytes()
    report_path = tmp_path / 'report.json'
    import builtins
    real_import = builtins.__import__
    def reject_ui_version_import(name, *args, **kwargs):
        if name == 'ui.pages.settings_page':
            raise ModuleNotFoundError('package check must not import GUI pages')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', reject_ui_version_import)
    result = recipe('launcher').package_check(report_path, path)
    assert result == 0, report_path.read_text(encoding='utf-8')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['hdf5']['time_axis_endpoint'] is endpoint
    assert report['hdf5']['time_step_s'] == pytest.approx(900e-9 / (501 - int(endpoint)))
    assert report['hdf5']['trace_metadata_verified'] is True
    assert report['hdf5']['missing_sensor_values'] is not height_present
    assert path.read_bytes() == before
