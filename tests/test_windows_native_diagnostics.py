"""Regression for Windows exceptions raised while other Python threads run."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows SEH regression")
def test_handled_windows_exception_does_not_walk_other_threads(tmp_path):
    script = textwrap.dedent(
        r"""
        import ctypes
        import faulthandler
        import sys
        import threading

        from core.observability import install_global_exception_hooks

        install_global_exception_hooks(sys.argv[1])
        ready = threading.Event()
        stop = threading.Event()

        def diagnostic_worker():
            ready.set()
            stop.wait(10)

        worker = threading.Thread(target=diagnostic_worker)
        worker.start()
        assert ready.wait(5)
        raise_exception = ctypes.windll.kernel32.RaiseException
        raise_exception.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
        try:
            raise_exception(0x8001010d, 0, 0, None)
        except OSError:
            pass
        finally:
            stop.set()
            worker.join()
            faulthandler.disable()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    log = (tmp_path / "native-crash.log").read_text(encoding="utf-8")
    assert "0x8001010d" in log
    assert "in diagnostic_worker" not in log
