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

from types import SimpleNamespace

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


def test_full_user_journey_processing_page(tmp_path, qapp, backend):
    """全用户旅程：加算法 → 选步骤改参 → 运行 → 禁用/删除/排序 →
    再运行 → 视图模式/统一色标/全部步骤开关 → 测线下拉文本。
    任一环异常即失败（处理页前端全覆盖）。"""
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

    project = backend.projects.create_project(tmp_path / 'proj', name='journey')
    pid = project.project_id
    rng = np.random.default_rng(21)
    matrix = np.cumsum(rng.normal(0, 0.4, size=(128, 256)),
                       axis=0).astype(np.float32)
    backend.projects.save_line_dataset(pid, 'L01', matrix, name='L01',
                                       length_m=50.0, time_window_ns=400.0)

    _PrevCmd, _Refresh, ProjectController, ProcessingPage = _imports()
    page = ProcessingPage()
    try:
        pc = ProjectController()
        pc._backend_controller = _BackendHost(backend)

        # ---- 测线列表到达：下拉文本必须可见（真机空白下拉回归） ----
        lines = list(backend.projects.list_lines(pid))
        page.set_lines(lines)
        assert page._line_combo.count() == 1
        assert page._line_combo.currentText() == 'L01', (
            f'测线下拉文本为空（真机空白下拉）：'
            f'{page._line_combo.currentText()!r}')

        # ---- 原始数据到位：输入卡 ----
        def _mk_bundle(tag, vmin=0.0):
            return SimpleNamespace(
                matrix=np.full((128, 256), float(tag), dtype=np.float32),
                vmin=float(vmin), vmax=float(tag), title='t', x_label='x',
                y_label='y', trace_axis_m=None, sample_axis=None,
                sample_axis_label='', trace_count=256, sample_count=128,
                trace_elevation_m=None, depth_axis_m=None)

        page.set_original_bundle(_mk_bundle(0.3))
        assert [c.key for c in page._result_grid.cards()] == ['input']

        # ---- 方法库到达 + 走真实添加路径（参数模板来自
        # parameter_schema 的 default——绕过它塞错参数会让运行 FAILED，
        # e2e 首轮即复现该用户风险路径）----
        page.set_methods([
            {'method_id': 'dewow', 'name': '零时校正',
             'display_name': '零时校正 (Dewow)',
             'parameter_schema': [{'name': 'window', 'default': 21}]},
            {'method_id': 'sec_gain', 'name': 'SEC增益',
             'display_name': 'SEC 增益 (AGC)',
             'parameter_schema': [{'name': 'gain_min', 'default': 1.0},
                                  {'name': 'gain_max', 'default': 6.0},
                                  {'name': 'power', 'default': 1.0}]},
        ])
        for mid in ('dewow', 'sec_gain'):
            page._add_method_to_pipeline(mid)
        chips = [s['method_id'] for s in page._pipeline_list.steps()]
        assert chips == ['dewow', 'sec_gain']
        assert page._pipeline_list.steps()[0]['params'] == {'window': 21}
        assert page._pipeline_list.steps()[1]['params'] == {
            'gain_min': 1.0, 'gain_max': 6.0, 'power': 1.0}
        assert len(page._chain_strip._steps) == 2

        # ---- 选中步骤 1 → 参数区跟随（有内容）----
        page._chain_strip.select_step(1)
        qapp.processEvents()
        assert page._chain_strip._list.currentRow() == 1

        # ---- 禁用步骤 0：链置灰（未运行不铺步骤卡——懒建原则）；
        # 后端提交时过滤禁用步（processing_service if step.enabled）----
        page._pipeline_list._toggle_enabled(0)
        qapp.processEvents()
        assert page._pipeline_list.steps()[0]['enabled'] is False
        assert page._chain_strip._steps[0]['enabled'] is False
        page._pipeline_list._toggle_enabled(0)
        qapp.processEvents()
        assert page._pipeline_list.steps()[0]['enabled'] is True

        # ---- 第一次运行（捕获 run_requested → 真提交）----
        jobs = []

        def _submit(payload):
            steps = [PipelineStep(s['method_id'], dict(s.get('params') or {}))
                     for s in payload['steps']]
            pipe = PipelineDefinition(name='journey', steps=tuple(steps))
            jobs.append(backend.submit_project_pipeline(
                pid, 'L01', pipe, result_name='journey-1'))

        page.run_requested.connect(_submit)
        page._on_run_clicked()
        assert jobs, '运行未提交'
        for job in jobs:
            assert backend.jobs.wait(job, timeout=120).status.name == 'COMPLETED'

        # ---- 刷新 → 结果按序铺卡 + 出图（批量预览宽松代数）----
        pc.artifact_preview_ready.connect(
            lambda aid, b: page.set_artifact_bundle(aid, b))
        got = {}
        pc.artifacts_updated.connect(lambda lid, a: got.update(a=list(a)))
        _Refresh(pc, pid, 'L01').execute()
        page.set_artifacts(list(got['a']))
        keys = [c.key for c in page._result_grid.cards()]
        assert keys == ['input', 'step:0', 'step:1']
        # 全部回填（骨架消失）
        for key in ('step:0', 'step:1'):
            aid = page._step_artifact_ids[int(key.split(':')[1])]
            _PrevCmd(pc, pid, 'L01', aid,
                                    pc._artifact_preview_generation,
                                    check_generation=False).execute()
            page.set_artifact_bundle(aid, ready_bundle(aid, pc, pid, 'L01'))
        for key in ('step:0', 'step:1'):
            assert page._result_grid.card(key).view._matrix is not None, key

        # ---- 改参数 → 脏提示；视图/开关交互 ----
        page._mark_results_stale()
        assert page._chain_strip._dirty_label.isVisibleTo(page._chain_strip)
        page._result_grid.set_shared_scale(True)
        page._result_grid.set_view_mode('single')
        page.set_colorbar_pref(True)
        page._result_grid.set_view_mode('all')
        page._on_expand_all_changed(False)
        qapp.processEvents()
        assert [c.key for c in page._result_grid.cards()] == ['input', 'step:1']
        page._on_expand_all_changed(True)

        # ---- 删除步骤 + 排序（链定义编辑）----
        # (1,0)=sec_gain 插到最前（(0,1) 是原位 noop 语义，返回 False）
        assert page._pipeline_list._move_step_to(1, 0) is True
        qapp.processEvents()
        page._pipeline_list._remove_step(1)
        assert len(page._pipeline_list.steps()) == 1

        # ---- 第二次运行（新组覆盖显示）----
        page._on_run_clicked()
        for job in jobs[1:]:
            assert backend.jobs.wait(job, timeout=120).status.name == 'COMPLETED'
        got2 = {}
        pc.artifacts_updated.connect(lambda lid, a: got2.update(a=list(a)))
        _Refresh(pc, pid, 'L01').execute()
        page.set_artifacts(list(got2['a']))
        assert page._result_grid.cards(), '第二次运行后无结果卡'

        # ---- 测线下拉在全程后仍有文本（refresh_lines 后空白回归）----
        assert page._line_combo.currentText() != '', '测线下拉文本为空'
    finally:
        _destroy_page(page, qapp)


def ready_bundle(aid, pc, pid, line_id):
    """真加载该成果的预览 bundle（与 _PreviewArtifactCommand 同路径）。"""
    pc._backend().projects.get_artifact_dataset_info(pid, line_id, aid)
    matrix, _s_idx, _t_idx = pc._backend().projects.read_artifact_window(
        pid, line_id, aid, max_samples=512, max_traces=512)
    from types import SimpleNamespace
    return SimpleNamespace(
        matrix=matrix, vmin=float(matrix.min()), vmax=float(matrix.max()),
        title=f'成果 {aid}', x_label='道数', y_label='走时 (ns)',
        trace_axis_m=None, sample_axis=None, sample_axis_label='',
        trace_count=int(matrix.shape[1]), sample_count=int(matrix.shape[0]),
        trace_elevation_m=None, depth_axis_m=None)
