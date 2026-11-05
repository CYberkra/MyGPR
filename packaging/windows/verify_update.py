"""Complete, signed MyGPR update acceptance on a disposable Windows CI runner."""
from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import ssl
import subprocess
import sys
import threading
import time

from scripts.publish_windows_update import make_appcast
from mygpr.infrastructure.updates.configuration import windows_build_version

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = 'https://localhost:18443'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wait_for(operation, description, timeout=60):
    deadline, last_error = time.monotonic() + timeout, ''
    while time.monotonic() < deadline:
        try:
            result = operation()
            if result:
                return result
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(0.2)
    raise TimeoutError(f'{description}: {last_error}')


def package(folder):
    metadata = json.loads((folder / 'build-provenance.json').read_text(encoding='utf-8'))
    installer = folder / 'assets' / metadata['installer_name']
    if digest(installer) != metadata['installer_sha256']:
        raise ValueError('Installer and build manifest do not match')
    return metadata, installer


def verified_inputs(folder, signature, tool):
    release, target = package(folder / 'release')
    seed, previous = package(folder / 'seed')
    if seed['version'] != release['version'] + 'a1' or seed['update_public_key'] != release['update_public_key']:
        raise ValueError('Unexpected seed version or public key')
    if len(base64.b64decode(signature, validate=True)) != 64:
        raise ValueError('Expected one Ed25519 installer signature')
    subprocess.run([str(tool), 'verify', '--public-key', release['update_public_key'], '--signature', signature,
                    str(target)], check=True, capture_output=True)
    subprocess.run(['git', 'merge-base', '--is-ancestor', release['application_commit'], 'HEAD'], cwd=ROOT, check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', release['application_commit'], 'HEAD'],
                                     cwd=ROOT, text=True).splitlines()
    allowed = ('.github/', 'tests/', 'docs/', 'packaging/windows/verify_', 'scripts/run_ci_regressions.py')
    if any(not name.startswith(allowed) for name in changed):
        raise ValueError('Application changed after the tested package was built: ' + ', '.join(changed))
    return release, target, seed, previous


class UpdateServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, installer, signature, version, seed_version, certificate, key):
        self.installer, self.signature = installer, signature
        self.version, self.seed_version = version, seed_version
        self.mode = 'valid'
        self.started, self.finished = threading.Event(), threading.Event()
        self.requests = []
        super().__init__(('localhost', 18443), UpdateHandler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
        self.socket = context.wrap_socket(self.socket, server_side=True)

    def select(self, mode):
        self.mode = mode
        self.started.clear()
        self.finished.clear()


class UpdateHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        server = self.server
        mode = server.mode
        server.requests.append({'path': self.path, 'mode': mode})
        if self.path.startswith('/appcast.xml'):
            if mode == 'offline':
                self.send_error(503, 'Acceptance: server unavailable')
                return
            signature = base64.b64encode(bytes(64)).decode() if mode == 'invalid-signature' else server.signature
            payload = make_appcast(version=server.seed_version if mode == 'current' else server.version,
                url=BASE_URL + '/setup.exe', notes_url=BASE_URL + '/notes', signature=signature,
                length=server.installer.stat().st_size)
            self.send_response(200)
            self.send_header('Content-Type', 'application/rss+xml; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.startswith('/setup.exe'):
            self.serve_installer(mode)
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(b'<html><body>MyGPR isolated update acceptance</body></html>')

    def serve_installer(self, mode):
        server = self.server
        self.send_response(200)
        self.send_header('Content-Length', str(server.installer.stat().st_size))
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        server.started.set()
        try:
            with server.installer.open('rb') as source:
                while block := source.read(64 * 1024):
                    self.wfile.write(block)
                    self.wfile.flush()
                    if mode == 'interrupted':
                        self.connection.shutdown(socket.SHUT_RDWR)
                        return
                    if mode == 'cancel':
                        time.sleep(0.02)
            server.finished.set()
        except (OSError, ssl.SSLError) as exc:
            server.requests.append({'download_ended': type(exc).__name__, 'mode': mode})


class Desktop:
    def __init__(self, process, output):
        from pywinauto import Application
        self.app = Application(backend='uia').connect(process=process.pid, timeout=60)
        self.process, self.output = process, output
        self.main = self.app.window(title_re='^MyGPR.*')
        self.main.wait('visible', timeout=90)

    def button(self, expression, *, native=False):
        windows = self.native_windows() if native else [self.main]
        for window in windows:
            for button in window.descendants(control_type='Button'):
                if (re.search(expression, button.window_text(), re.I)
                        and button.is_visible() and button.is_enabled()):
                    return button
        return None

    def native_windows(self):
        return [window for window in self.app.windows() if window.is_visible()
                and not window.class_name().startswith('Qt')]

    def about(self):
        from pywinauto.mouse import scroll
        self.main.set_focus()
        wait_for(lambda: self.button('^设置$'), 'settings navigation').click_input()
        rectangle = self.main.rectangle()
        scroll(coords=(rectangle.right - 60, rectangle.bottom - 180), wheel_dist=-35)
        wait_for(lambda: self.button('^检查更新$'), 'enabled update entry')

    def check(self):
        wait_for(lambda: self.button('^检查更新$'), 'updater idle').click_input()

    def status(self, text):
        return any(text in control.window_text() for control in self.main.descendants(control_type='Text'))

    def close_native(self):
        for window in self.native_windows():
            window.close()
        wait_for(lambda: self.button('^检查更新$'), 'updater returned to idle')

    def capture(self, name):
        self.main.capture_as_image().save(self.output / f'{name}-main.png')
        for index, window in enumerate(self.native_windows()):
            window.capture_as_image().save(self.output / f'{name}-native-{index}.png')

    def diagnostics(self):
        self.main.print_control_identifiers(filename=str(self.output / 'main-controls.txt'))
        for index, window in enumerate(self.native_windows()):
            window.print_control_identifiers(filename=str(self.output / f'native-{index}-controls.txt'))
        self.capture('diagnostic')


def synthetic_source(path):
    import h5py
    import numpy as np
    waveform = np.arange(32 * 65, dtype=np.float32).reshape(32, 65) / 1000
    with h5py.File(path, 'w') as handle:
        handle.attrs.update(samples_per_trace=65, time_window_ns=500., store_component='fdata',
                            committed_trace_count=32, simulation=True, data_source='SYNTHETIC_UPDATE_ACCEPTANCE',
                            acquisition_id='6b344aa1-ac15-4238-943a-0c7c2d165217', time_axis_endpoint=False)
        handle.create_dataset('traces', data=waveform)
        for name in ('lon', 'lat', 'alt', 'height', 'roll', 'pitch', 'yaw', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'):
            handle.create_dataset('trace_meta/' + name, data=np.full(32, np.nan))
        handle.create_dataset('trace_meta/timestamp_ms', data=np.arange(32) * 25. + 1792000000000.)
        handle.create_dataset('trace_meta/trace_id', data=np.arange(32, dtype=np.uint64))
    return waveform.T


def seed_project(folder, source):
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(folder, name='更新验收项目')
        job = backend.submit_batch_line_import(project.project_id, [str(source)], dielectric_constant=7.5)
        imported = backend.jobs.wait(job, timeout=60)
        assert imported.status.value == 'completed', imported.error_message
        line = backend.projects.list_lines(project.project_id)[0]
        processed = backend.jobs.wait(backend.submit_project_pipeline(project.project_id, line.line_id,
            PipelineDefinition(name='更新前成果', steps=(PipelineStep('dewow', {'window': 7}),))), timeout=60)
        assert processed.status.value == 'completed', processed.error_message
        return line.line_id, processed.result.artifact_id
    finally:
        backend.shutdown()


def scientific_hashes(project):
    return {str(path.relative_to(project)): digest(path) for path in project.rglob('*')
            if path.is_file() and path.suffix.lower() in ('.h5', '.hdf5', '.npy', '.npz')}


def exercise_rejections(ui, server, record):
    for mode, expected in (('offline', '失败'), ('current', '最新版本')):
        server.select(mode)
        ui.check()
        wait_for(lambda: ui.status(expected), mode)
        ui.capture(mode)
        ui.close_native()
        assert ui.process.poll() is None
        record[mode] = 'passed'
    server.select('valid')
    ui.check()
    wait_for(lambda: ui.button('稍后|以后|later', native=True), 'remind later').click_input()
    wait_for(lambda: ui.button('^检查更新$'), 'postponed update')
    record['postpone'] = 'passed'
    ui.check()
    wait_for(lambda: ui.button('跳过|skip', native=True), 'skip version').click_input()
    wait_for(lambda: ui.button('^检查更新$'), 'skipped update')
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\MyGPR Project\MyGPR\WinSparkle') as key:
        skipped, _ = winreg.QueryValueEx(key, 'SkipThisVersion')
    assert skipped == windows_build_version(server.version)
    record['skip-version-persisted'] = 'passed'
    for mode in ('cancel', 'interrupted', 'invalid-signature'):
        server.select(mode)
        ui.check()
        wait_for(lambda: ui.button('安装|Install', native=True), mode + ' install button').click_input()
        assert server.started.wait(30), mode + ' download did not start'
        if mode == 'cancel':
            ui.capture(mode)
        else:
            wait_for(lambda: ui.status('失败'), mode + ' rejected', timeout=120)
            ui.capture(mode)
        ui.close_native()
        assert ui.process.poll() is None, mode + ' closed the old application'
        record[mode] = 'passed'


def install_and_relaunch(ui, server, executable, output, version, record):
    from pywinauto import Desktop as WindowsDesktop
    server.select('valid')
    ui.check()
    wait_for(lambda: ui.button('安装|Install', native=True), 'download signed update').click_input()
    assert server.finished.wait(120), 'complete installer was not served'
    install = wait_for(lambda: ui.button('安装|Install', native=True), 'verified installer ready', timeout=120)
    ui.capture('verified-ready')
    install.click_input()
    assert ui.process.wait(timeout=60) == 0, 'application did not close normally'

    def restarted():
        for window in WindowsDesktop(backend='uia').windows(title_re='^MyGPR.*', visible_only=True):
            if window.process_id() != ui.process.pid:
                return window
        return None

    wrapper = wait_for(restarted, 'installed application relaunch', timeout=180)
    from pywinauto import Application
    app = Application(backend='uia').connect(process=wrapper.process_id())
    window = app.window(handle=wrapper.handle)
    window.wait('enabled', timeout=90)
    window.capture_as_image().save(output / 'restarted.png')
    installed = json.loads((executable.parent / '_internal/windows-build.json').read_text(encoding='utf-8'))
    assert installed['version'] == version
    live = Desktop.__new__(Desktop)
    live.app, live.main, live.output = app, window, output
    live.about()
    window.child_window(title=version, control_type='Text').wait('exists', timeout=30)
    live.capture('installed-version')
    record['restarted-normal-exit-code'] = close_with_zero_exit(window, app)
    record['signed-download-install-relaunch'] = 'passed'


def close_with_zero_exit(window, app):
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.GetExitCodeProcess.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, app.process)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        window.close()
        app.wait_for_process_exit(timeout=60)
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        assert code.value == 0, f'Restarted application exited abnormally: 0x{code.value:08X}'
        return code.value
    finally:
        kernel.CloseHandle(handle)


def verify_preserved(project, source, expected, hashes, line_id, artifact_id, settings):
    import numpy as np
    from mygpr.interfaces.backend import MyGPRBackend
    assert scientific_hashes(project) == hashes, 'project data or result changed during update'
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        opened = backend.projects.open_project(project)
        data = backend.projects.read_dataset(opened.project_id, line_id)
        np.testing.assert_array_equal(np.asarray(data.data), expected)
        assert any(item.artifact_id == artifact_id for item in backend.projects.list_artifacts(opened.project_id, line_id))
    finally:
        backend.shutdown()
    saved = json.loads(settings.read_text(encoding='utf-8'))
    assert saved['theme'] == '深色主题' and saved['default_dielectric'] == 7.5
    assert str(project) in saved['recent_projects']
    assert saved['update_channel'] == 'stable' and saved['update_automatic'] is False
    assert source.is_file()


def run_acceptance(inputs, output, signature, certificate, key, tool):
    if sys.platform != 'win32' or os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Complete installation acceptance requires a disposable GitHub Windows runner')
    spec = importlib.util.spec_from_file_location('install_qa', ROOT / 'packaging/windows/verify_install.py')
    install_qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(install_qa)
    install_qa.ensure_clean_installation()
    release, target, seed, previous = verified_inputs(inputs, signature, tool)
    output.mkdir(parents=True, exist_ok=True)
    source, project = output / 'original.h5', output / 'project'
    expected = synthetic_source(source)
    original_hash = digest(source)
    line_id, artifact_id = seed_project(project, source)
    hashes = scientific_hashes(project)
    from ui.constants import SETTINGS_FILE
    settings = Path(SETTINGS_FILE)
    if settings.exists():
        raise RuntimeError('Disposable runner already has MyGPR settings')
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({'theme': '深色主题', 'default_dielectric': 7.5, 'recent_projects': [str(project)],
        'update_automatic': False, 'update_channel': 'preview'}, ensure_ascii=False), encoding='utf-8')
    install = output / 'installed'
    subprocess.run([str(previous), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/SP-', '/NORESTART', '/CURRENTUSER',
                    f'/DIR={install}'], check=True, timeout=120)
    server = UpdateServer(target, signature, release['version'], seed['version'], certificate, key)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    executable = install / 'MyGPR.exe'
    environment = os.environ.copy()
    environment.update(QT_QPA_PLATFORM='windows:darkmode=0', QT_ACCESSIBILITY='1')
    environment.pop('MYGPR_SETTINGS_FILE', None)
    process = subprocess.Popen([str(executable)], cwd=output, env=environment)
    evidence = {'version': release['version'], 'application_commit': release['application_commit'],
                'installer_sha256': digest(target), 'checks': {}, 'status': 'failed'}
    ui = None
    try:
        ui = Desktop(process, output)
        ui.about()
        exercise_rejections(ui, server, evidence['checks'])
        install_and_relaunch(ui, server, executable, output, release['version'], evidence['checks'])
        package_report = output / 'installed-package.json'
        subprocess.run([str(executable), '--package-check', str(package_report), '--hdf5', str(source)],
                       check=True, timeout=120)
        install_qa.verify_package_report(json.loads(package_report.read_text(encoding='utf-8')), release['version'])
        verify_preserved(project, source, expected, hashes, line_id, artifact_id, settings)
        assert digest(source) == original_hash
        subprocess.run([str(install / 'unins000.exe'), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'],
                       check=True, timeout=120)
        evidence['checks']['settings-project-result-and-original-preserved'] = 'passed'
        evidence['status'] = 'passed'
    except Exception as exc:
        evidence['error'] = repr(exc)
        if ui is not None and process.poll() is None:
            ui.diagnostics()
        raise
    finally:
        evidence['http_requests'] = server.requests
        (output / 'update-acceptance.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
