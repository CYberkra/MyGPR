"""The diff gate distinguishes invalid evidence from changes outside its scope."""
import json
import subprocess

from scripts import check_diff_coverage as gate


def invoke(tmp_path, changed, coverage):
    changed_path = tmp_path / 'changes.json'
    coverage_path = tmp_path / 'coverage.json'
    changed_path.write_text(json.dumps(changed), encoding='utf-8')
    coverage_path.write_text(json.dumps({'files': coverage}), encoding='utf-8')
    return gate.main([str(coverage_path), '--changed-lines', str(changed_path), '--require-diff'])


def test_packaging_changes_do_not_require_backend_coverage(tmp_path):
    assert invoke(tmp_path, {'packaging/windows/launcher.py': [1, 2]}, {}) == 0


def test_changed_backend_requires_coverage_evidence(tmp_path):
    assert invoke(tmp_path, {'core/new_reader.py': [1]}, {}) == 1


def test_windows_coverage_paths_match_git_paths(tmp_path):
    assert invoke(tmp_path, {'core/reader.py': [1]},
                  {r'core\reader.py': {'executed_lines': [1], 'missing_lines': []}}) == 0


def test_windows_paths_do_not_bypass_uncovered_line_threshold(tmp_path):
    assert invoke(tmp_path, {'core/reader.py': [1, 2]},
                  {r'core\reader.py': {'executed_lines': [1], 'missing_lines': [2]}}) == 1
    report = json.loads((tmp_path / 'diff-coverage-report.json').read_text(encoding='utf-8'))
    assert report['total'] == 2 and report['covered'] == 1


def test_uncovered_backend_fails_threshold(tmp_path):
    assert invoke(tmp_path, {'core/reader.py': [1, 2]},
                  {'core/reader.py': {'executed_lines': [1], 'missing_lines': [2]}}) == 1


def test_covered_backend_and_comment_only_pass(tmp_path):
    assert invoke(tmp_path, {'core/reader.py': [1, 2]},
                  {'core/reader.py': {'executed_lines': [1], 'missing_lines': []}}) == 0
    assert invoke(tmp_path, {'core/reader.py': [2]},
                  {'core/reader.py': {'executed_lines': [1], 'missing_lines': []}}) == 0


def test_invalid_base_fails(tmp_path, monkeypatch):
    coverage_path = tmp_path / 'coverage.json'
    coverage_path.write_text('{"files": {}}', encoding='utf-8')
    monkeypatch.setattr(gate, '_ref_exists', lambda ref: False)
    assert gate.main([str(coverage_path), '--base', 'missing', '--require-diff']) == 1


def test_diff_error_fails_instead_of_becoming_empty_map(tmp_path, monkeypatch):
    coverage_path = tmp_path / 'coverage.json'
    coverage_path.write_text('{"files": {}}', encoding='utf-8')
    monkeypatch.setattr(gate, '_ref_exists', lambda ref: True)
    monkeypatch.setattr(gate.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess(a, 128, '', 'bad revision'))
    assert gate.main([str(coverage_path), '--head', 'missing', '--require-diff']) == 1


def test_unicode_backend_filename_is_checked_with_default_git_quoting(tmp_path, monkeypatch):
    def git(*arguments):
        return subprocess.run(['git', *arguments], cwd=tmp_path, check=True,
                              capture_output=True, text=True, encoding='utf-8').stdout.strip()

    git('init')
    git('config', 'user.name', 'Coverage test')
    git('config', 'user.email', 'coverage@example.invalid')
    git('config', 'core.quotepath', 'true')
    path = tmp_path / 'core/中文模块.py'
    path.parent.mkdir()
    path.write_text('value = 1\n', encoding='utf-8')
    git('add', '.')
    git('commit', '-m', 'base')
    base = git('rev-parse', 'HEAD')
    path.write_text('value = 2\n', encoding='utf-8')
    git('commit', '-am', 'change')
    with monkeypatch.context() as patch:
        patch.setattr(gate, 'ROOT', tmp_path)
        assert gate._git_changed_lines(base, 'HEAD') == {'core/中文模块.py': {1}}
    assert invoke(tmp_path, {'core/中文模块.py': [1]}, {}) == 1
