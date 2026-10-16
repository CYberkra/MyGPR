"""Release QA contracts, without launching or changing a Windows installation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest


@pytest.fixture
def verifier():
    path = Path(__file__).parents[1] / 'packaging/windows/verify_install.py'
    spec = importlib.util.spec_from_file_location('installer_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_version_is_required(verifier):
    with pytest.raises(SystemExit) as error:
        verifier.parse_args(['setup.exe', '--hdf5', 'line.h5', '--output', 'qa'])
    assert error.value.code == 2


@pytest.mark.parametrize('version', ['0.11.0', '9.8.7'])
def test_expected_version_is_explicit_and_not_release_specific(verifier, version):
    args = verifier.parse_args([
        'setup.exe', '--hdf5', 'line.h5', '--output', 'qa',
        '--expected-version', version,
    ])
    assert args.expected_version == version


def package_report(version):
    return {
        'frozen': True,
        'status': 'passed',
        'versions': {name: version for name in ('distribution', 'report', 'settings')},
        'hdf5': {'source_unchanged': True},
    }


def test_package_version_and_source_are_verified(verifier):
    verifier.verify_package_report(package_report('0.11.0'), '0.11.0')


@pytest.mark.parametrize('problem', ['old_version', 'missing_version', 'changed_source'])
def test_incomplete_or_wrong_package_evidence_is_rejected(verifier, problem):
    report = package_report('0.11.0')
    if problem == 'old_version':
        report['versions']['settings'] = '0.10.0'
    elif problem == 'missing_version':
        del report['versions']['report']
    else:
        report['hdf5']['source_unchanged'] = False
    with pytest.raises(AssertionError):
        verifier.verify_package_report(report, '0.11.0')


SMOKE_PAGES = (
    'projectInterface', 'processingInterface', 'interpretationInterface',
    'spatialInterface', 'deliveryInterface', 'jobsInterface', 'settingsInterface',
)


def write_smoke_images(folder):
    for name in [*(f'page_{page}.png' for page in SMOKE_PAGES), 'window_overall.png']:
        Image.new('RGB', (16, 16)).save(folder / name)


def test_seven_pages_plus_window_are_accepted(verifier, tmp_path):
    write_smoke_images(tmp_path)
    names = verifier.verify_smoke_screenshots(tmp_path)
    assert len(names) == 8
    assert 'page_homeInterface.png' not in names


@pytest.mark.parametrize('problem', ['missing_page', 'legacy_extra', 'empty_image'])
def test_smoke_file_count_cannot_hide_wrong_page_or_invalid_image(verifier, tmp_path, problem):
    write_smoke_images(tmp_path)
    if problem == 'missing_page':
        (tmp_path / 'page_processingInterface.png').unlink()
        Image.new('RGB', (16, 16)).save(tmp_path / 'unrelated.png')
    elif problem == 'legacy_extra':
        Image.new('RGB', (16, 16)).save(tmp_path / 'page_homeInterface.png')
    else:
        (tmp_path / 'page_projectInterface.png').write_bytes(b'')
    with pytest.raises(AssertionError):
        verifier.verify_smoke_screenshots(tmp_path)


def fake_registry():
    calls = []

    def open_key(hive, key, _reserved, access):
        calls.append((hive, key, access))
        raise FileNotFoundError(key)

    registry = SimpleNamespace(
        HKEY_CURRENT_USER='current-user', HKEY_LOCAL_MACHINE='machine',
        KEY_WOW64_64KEY=256, KEY_WOW64_32KEY=512, KEY_READ=1,
        OpenKey=open_key,
    )
    return registry, calls


def test_install_guard_checks_both_hives_and_registry_views(verifier):
    registry, calls = fake_registry()
    verifier.ensure_clean_installation(registry)
    assert {(hive, access) for hive, _key, access in calls} == {
        ('current-user', 257), ('current-user', 513),
        ('machine', 257), ('machine', 513),
    }


def test_existing_installation_is_not_replaced(verifier):
    class ExistingKey:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    registry, _calls = fake_registry()
    registry.OpenKey = lambda *_args: ExistingKey()
    with pytest.raises(RuntimeError, match='Existing MyGPR installer registration'):
        verifier.ensure_clean_installation(registry)
