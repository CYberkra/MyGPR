# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MyGPR Application
"""

block_cipher = None

a = Analysis(
    ['app_qt.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 主题扩展文件
        ('assets/styles_extra_light.qss', 'assets'),
        ('assets/styles_extra_dark.qss', 'assets'),
        ('ui/qss/app_polish_template.qss', 'ui/qss'),
        # 配置文件
        ('assets/theme_config.json', 'assets'),
        ('VERSION', '.'),
        # 启动脚本
        ('启动MyGPR.bat', '.'),
    ],
    hiddenimports=[
        # PyQt6 核心
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtSvg',
        'PyQt6.QtOpenGL',
        
        # 科学计算
        'numpy',
        'numpy.core._multiarray_umath',
        'numpy.lib.format',
        'scipy',
        'scipy.special._ufuncs_cxx',
        'scipy.io',
        'scipy.signal',
        'scipy.interpolate',
        'scipy.optimize',
        'pywt',
        
        # Matplotlib
        'matplotlib',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
        'matplotlib.pyplot',
        
        # Pandas
        'pandas',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.interval',
        
        # GUI 扩展
        'qfluentwidgets',
        'pyqtgraph',
        'pyqtgraph.opengl',
        'OpenGL',
        'OpenGL.GL',
        
        # 项目模块
        'core.gpr_io',
        'core.methods_registry',
        'core.processing_engine',
        'core.shared_data_state',
        'core.shared_data_model',
        'ui.shared_data_qt_adapter',
        'ui.report_export_controller',
        'ui.processing_lineage_controller',
        'ui.bscan_interaction_controller',
        'ui.autotune_sync_controller',
        'ui.processing_worker_controller',
        'core.autotune_background_runner',
        'core.autotune_scoring_weights',
        'core.autotune_recipe',
        'core.autotune_workflow_planner',
        'core.autotune_recipe_runner',
        'core.app_paths',
        'core.theme_manager',
        'core.workflow_data',
        'core.workflow_executor',
        'core.workflow_template_manager',
        'core.favorites_manager',
        'ui.gui_basic_flow',
        'ui.gui_advanced_settings',
        'ui.gui_quality_log',
        'ui.georef3d_results_page',
        'ui.gui_method_browser',
        'ui.gui_param_editor',
        'ui.gui_workbench',
        'ui.gui_base',
        'ui.loading_dialog',
        'read_file_data',
        
        # PythonModule 处理模块
        'PythonModule',
        'PythonModule.dewow',
        'PythonModule.compensatingGain',
        'PythonModule.subtracting_average_2D',
        'PythonModule.median_background_2D',
        'PythonModule.sec_gain',
        'PythonModule.fk_filter',
        'PythonModule.hankel_svd',
        'PythonModule.svd_subspace',
        'PythonModule.wavelet_svd',
        'PythonModule.svd_background',
        'PythonModule.stolt_migration',
        'PythonModule.time_to_depth',
        'PythonModule.sliding_average',
        'PythonModule.ccbs_filter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 'tkinter',  # 保留，matplotlib 可能需要
        'test',
        # 'unittest',  # pyparsing 需要
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MyGPR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 设置为 True 可以看到控制台输出用于调试
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico' if __import__('os').path.exists('app_icon.ico') else None,
)
