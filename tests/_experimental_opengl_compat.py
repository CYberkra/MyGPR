"""Match PyOpenGL's WGL implementation to Qt's selected Windows renderer."""
import ctypes
import sys


def align_windows_opengl() -> None:
    if sys.platform != 'win32':
        return

    from PyQt6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext

    if QGuiApplication.platformName() != 'windows':
        return

    previous = QOpenGLContext.currentContext()
    previous_surface = previous.surface() if previous is not None else None
    surface = QOffscreenSurface()
    surface.create()
    context = QOpenGLContext()
    if not surface.isValid() or not context.create() or not context.makeCurrent(surface):
        raise RuntimeError('无法创建 OpenGL 探测上下文')
    try:
        kernel = ctypes.windll.kernel32
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        kernel.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        kernel.GetProcAddress.restype = ctypes.c_void_p
        module = kernel.GetModuleHandleW('opengl32sw.dll')
        address = kernel.GetProcAddress(module, b'wglGetCurrentContext') if module else None
        software_current = address and ctypes.WINFUNCTYPE(ctypes.c_void_p)(address)()
        if not software_current:
            return
        if 'OpenGL.GL' in sys.modules:
            raise RuntimeError('OpenGL 已在选择软件渲染器之前加载')
        from OpenGL import platform

        library = ctypes.WinDLL('opengl32sw.dll', handle=module)
        implementation = platform.PLATFORM
        implementation.GL = library
        implementation.__dict__.pop('OpenGL', None)
        implementation.__dict__.pop('WGL', None)
        implementation.__dict__.pop('GetCurrentContext', None)
        implementation.__dict__.pop('CurrentContextIsValid', None)
        implementation.__dict__.pop('getExtensionProcedure', None)
        implementation.install(platform.__dict__)
    finally:
        context.doneCurrent()
        if previous is not None and previous_surface is not None:
            previous.makeCurrent(previous_surface)
