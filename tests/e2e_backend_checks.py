# -*- coding: utf-8 -*-
"""端到端回归：真后端 → 真控制器命令 → 处理页结果卡。

2026-09-27 真机「运行完没反应」真根因：``backend.projects.list_artifacts``
返回的 ``ProjectArtifact.manifest`` 是 **types.MappingProxyType**（后端
冻结数据的只读视图），页面 ``isinstance(manifest, dict)`` 判假 →
run_group 永远取空 → 结果卡一个不铺。此前的页面级测试喂的是手工 dict
（形状失真），全绿但真机全废——本文件改用**真实后端对象**走完整链：

    真运行（submit_project_pipeline + jobs.wait）
    → 真刷新（_RefreshArtifactsCommand.execute → artifacts_updated）
    → 页面（set_artifacts → auto-open → 结果网格）
    → 真预览（_PreviewArtifactCommand.execute → artifact_preview_ready）
    → 卡片出图（set_artifact_bundle → view._matrix 非空）

页面直调接缝（set_artifact_bundle 直喂）从此只作辅助，形状类 bug 以
本文件为准。
"""
import gc

import numpy as np
import pytest

# 注意：本文件的**全部实质 import（含 ui.controllers.project_controller
# ——其链上会拉 mygpr 后端模块与 h5py/DLL）都延迟到测试函数内**：
# 模块级 import 会在 pytest 收集阶段改变全局进程状态，与前序重量级
# GUI 测试交互出 access violation（2026-09-27 全量实锤：本文件存在
# 即崩 test_bscan_container:413，--ignore 本文件即全绿；两次对照成立）。


@pytest.fixture
def backend():
    from mygpr.interfaces.backend import MyGPRBackend
    b = MyGPRBackend.create_default(max_workers=1)
    yield b
    b.shutdown()


class _BackendHost:
    """模拟 BackendController 的 ``.backend`` 挂点（控制器取后端的口）。"""

    def __init__(self, backend):
        self.backend = backend


def _run_real_pipeline(backend, root: str) -> str:
    """真建项目/测线/运行（与桌面应用同一条提交路径）。"""
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    project = backend.projects.create_project(root, name='e2e')
    pid = project.project_id
    rng = np.random.default_rng(7)
    matrix = np.cumsum(rng.normal(0, 0.4, size=(128, 256)),
                       axis=0).astype(np.float32)
    backend.projects.save_line_dataset(
        pid, 'L01', matrix, name='e2e line',
        length_m=50.0, time_window_ns=400.0)
    pipeline = PipelineDefinition(
        name='e2e',
        steps=(
            PipelineStep('dewow', {'window': 23}),
            PipelineStep('sec_gain', {'gain_min': 1.0, 'gain_max': 6.0,
                                      'power': 1.0}),
        ),
    )
    job_id = backend.submit_project_pipeline(
        pid, 'L01', pipeline, result_name='e2e-run')
    snap = backend.jobs.wait(job_id, timeout=120)
    assert snap.status.name == 'COMPLETED', snap.error_message
    return pid


def _imports():
    from ui.controllers.project_controller import (
        _PreviewArtifactCommand, _RefreshArtifactsCommand, ProjectController)
    from ui.pages.processing_page import ProcessingPage
    return _PreviewArtifactCommand, _RefreshArtifactsCommand, ProjectController, ProcessingPage


def _destroy_page(page, qapp):
    """显式销毁：e2e 的 page 持有真后端线程接触过的 C++ 对象，若交给
    延迟 GC，会在**后续**测试运行中途析构（qfluentwidgets/pyqtgraph
    场景图并发访问）→ access violation（2026-09-27 全量实锤）。
    deleteLater + 立即处理事件 + 强制分代回收，把析构钉在本测试内。"""
    page.deleteLater()
    qapp.processEvents()
    qapp.processEvents()
    del page
    gc.collect()
    qapp.processEvents()


def test_run_to_result_cards_over_real_backend(tmp_path, qapp, backend):
    pid = _run_real_pipeline(backend, str(tmp_path / 'proj'))
    _PrevCmd, _Refresh, ProjectController, ProcessingPage = _imports()
    page = ProcessingPage()
    try:
        pc = ProjectController()
        pc._backend_controller = _BackendHost(backend)

        # ---- 1) 真刷新命令 → artifacts_updated（与 _RefreshArtifactsCommand
        #         被 run_command 线程执行的同一 execute）----
        captured = {}
        pc.artifacts_updated.connect(
            lambda line_id, arts: captured.update(line_id=line_id, arts=arts))
        _Refresh(pc, pid, 'L01').execute()
        assert captured, '刷新命令未发 artifacts_updated'
        arts = captured['arts']
        assert len(arts) == 2                       # 1 intermediate + 1 final

        # ---- 2) 真 manifest 形状能被页面分组（mappingproxy 回归锁）----
        assert any(type(a.manifest).__name__ == 'mappingproxy' for a in arts)
        params = page._run_group_params(arts[0])
        assert str(params.get('run_group_id') or ''), (
            'mappingproxy manifest 被页面拒收（真机「运行完没反应」根因）')

        # ---- 3) 页面吃真成果列表 → 结果网格按序铺卡 ----
        page.set_artifacts(list(arts))
        keys = [c.key for c in page._result_grid.cards()]
        assert keys == ['input', 'step:0', 'step:1']
        titles = [c.title_label.text() for c in page._result_grid.cards()]
        assert 'dewow' in titles[1] and 'sec_gain' in titles[2]
        assert len(page._result_grid.cards()) == 3

        # ---- 4) 真预览命令 → 真 bundle → 卡片出图 ----
        intermediate = next(a for a in arts
                            if a.manifest.get('artifact_kind') == 'intermediate')
        ready, failed = {}, []
        pc.artifact_preview_ready.connect(
            lambda aid, bundle: ready.update(aid=aid, bundle=bundle))
        pc.preview_failed.connect(failed.append)
        # 代数守卫：命令回包要求与控制器当前代数一致（真实提交经
        # preview_artifact 自增；这里取当前值——0）
        _PrevCmd(pc, pid, 'L01', intermediate.artifact_id,
                 pc._artifact_preview_generation).execute()
        assert ready and not failed, f'预览未回包/失败：ready={ready} failed={failed}'
        page.set_artifact_bundle(ready['aid'], ready['bundle'])
        card = page._result_grid.card('step:0')
        assert card is not None and card.view is not None
        assert card.view._matrix is not None        # 出图（非骨架）
    finally:
        _destroy_page(page, qapp)


def test_auto_open_tabs_from_real_manifest(tmp_path, qapp, backend):
    """步骤 tab（旧模型并存）同样要被真 manifest 点亮。"""
    pid = _run_real_pipeline(backend, str(tmp_path / 'proj'))
    _PrevCmd, _Refresh, ProjectController, ProcessingPage = _imports()
    page = ProcessingPage()
    try:
        arts = list(backend.projects.list_artifacts(pid, 'L01'))
        page.set_artifacts(arts)
        sources = [s['key'] for s in page._preview_sources]
        assert sources and sources != ['original']   # 不再只有原始数据
        artifact_keys = [k for k in sources if k.startswith('artifact:')]
        assert len(artifact_keys) == 2               # 1 中间 + 1 最终
    finally:
        _destroy_page(page, qapp)


def test_history_run_replay_on_line_open(tmp_path, qapp, backend):
    """真机反馈场景：打开项目/切测线后「下面有历史结果、上面链为空」
    + 骨架挂死。真后端两条测线各跑一条链，模拟真实时序：
    成果列表先到（组切换 → 链回显）→ 批量预览（宽松代数）→ 全部回填。
    """
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    project = backend.projects.create_project(tmp_path / 'proj', name='e2e2')
    pid = project.project_id
    rng = np.random.default_rng(11)
    for line_id, seed in (('L01', 1), ('L02', 2)):
        matrix = np.cumsum(rng.normal(0, 0.4, size=(96, 200)),
                           axis=0).astype(np.float32)
        backend.projects.save_line_dataset(
            pid, line_id, matrix, name=line_id,
            length_m=40.0, time_window_ns=400.0)
    pipelines = {
        'L01': PipelineDefinition(name='r1', steps=(
            PipelineStep('dewow', {'window': 23}),
            PipelineStep('sec_gain', {'gain_min': 1.0, 'gain_max': 6.0,
                                      'power': 1.0}))),
        'L02': PipelineDefinition(name='r2', steps=(
            PipelineStep('subtracting_average_2D', {'ntraces': 16}),)),
    }
    for line_id, pipeline in pipelines.items():
        job = backend.submit_project_pipeline(pid, line_id, pipeline,
                                              result_name=f'run-{line_id}')
        snap = backend.jobs.wait(job, timeout=120)
        assert snap.status.name == 'COMPLETED', snap.error_message

    _PrevCmd, _Refresh, ProjectController, ProcessingPage = _imports()
    page = ProcessingPage()
    try:
        pc = ProjectController()
        pc._backend_controller = _BackendHost(backend)
        # 页面预览请求 → 真加载命令（宽松代数，模拟协调器接线）
        def _serve(artifact_id):
            _PrevCmd(pc, pid, 'L02', artifact_id,
                     pc._artifact_preview_generation,
                     check_generation=False).execute()
        ready = {}
        pc.artifact_preview_ready.connect(
            lambda aid, bundle: (ready.update(aid=aid, bundle=bundle),
                                 page.set_artifact_bundle(aid, bundle)))
        page.artifact_preview_requested.connect(_serve)

        # ---- 用户切到 L02：成果列表到达 ----
        arts_l02 = list(backend.projects.list_artifacts(pid, 'L02'))
        page.set_artifacts(arts_l02)
        chips = [s['method_id'] for s in page._pipeline_list.steps()]
        assert chips == ['subtracting_average_2D'], (
            f'链未回显历史 run：{chips}')
        keys = [c.key for c in page._result_grid.cards()]
        assert keys == ['input', 'step:0']
        qapp.processEvents()
        assert ready, '骨架未回填（批量预览被代数守卫丢弃）'
        card = page._result_grid.card('step:0')
        assert card.view._matrix is not None            # 出图

        # ---- 切回 L01：链跟随新测线的历史 run（2 步）----
        arts_l01 = list(backend.projects.list_artifacts(pid, 'L01'))
        page.artifact_preview_requested.disconnect(_serve)

        def _serve_l01(artifact_id):
            _PrevCmd(pc, pid, 'L01', artifact_id,
                     pc._artifact_preview_generation,
                     check_generation=False).execute()
        page.artifact_preview_requested.connect(_serve_l01)
        page.set_artifacts(arts_l01)
        chips = [s['method_id'] for s in page._pipeline_list.steps()]
        assert chips == ['dewow', 'sec_gain'], f'切线后链未跟随：{chips}'
        keys = [c.key for c in page._result_grid.cards()]
        assert keys == ['input', 'step:0', 'step:1']
        qapp.processEvents()
        for key in ('step:0', 'step:1'):
            card = page._result_grid.card(key)
            assert card.view._matrix is not None, f'{key} 骨架未回填'
    finally:
        _destroy_page(page, qapp)
