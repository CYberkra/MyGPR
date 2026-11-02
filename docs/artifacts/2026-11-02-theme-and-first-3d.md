# 主题切换与首次三维窗口验证

范围：MyGPR 当前源码的换肤耗时和 Windows 首次打开三维视图时的整窗消失。
保留原界面结构、处理算法、数据分辨率及文件格式；未打包 EXE。

## 原因与改动

- 完整原生 UI（1190 个 QWidget，独立设置、控制器禁用）的主题槽基线为深色 1861 ms、浅色 1831 ms。
  cProfile 显示应用级 QSS 替换约 1031 ms，Fluent 样式全量刷新两次约 700 ms。
- 原生表格/列表/表头/页签的应用 QSS 改为稳定的 palette 规则；显式同步匹配控件的 palette 并局部 polish。
  单纯 QApplication.setPalette 在已存在 QSS 视图上会保留旧颜色，已用独立原生窗口复现，因此没有只改 palette 后省略刷新。
- Fluent 采用库提供的 lazy 换肤路径，隐藏控件在绘制前更新。强调色未变化时不重复触发全量样式刷新。
- 独立审查发现表格角按钮不属于 QAbstractItemView，已补入 QTableCornerButton 刷新，并用真实渲染像素断言深浅往返颜色。
- 首次动态添加 QOpenGLWidget 时，Qt 将已显示的 Windows 原生窗口从 RasterSurface 改为 OpenGLSurface。
  基线确实记录了 Hide / WinIdChange / Show 与 HWND 改变；AA_ShareOpenGLContexts 的独立对照未解决此问题。
  现于主窗口显示前预选 OpenGLSurface，保留 GL 模块和上下文的延迟创建。
  机制参考 [Qt QOpenGLWidget 官方文档](https://doc.qt.io/qt-6/qopenglwidget.html#limitations-and-other-considerations)。

## 结果与边界

- 同一完整原生界面的主题槽后测约 291～504 ms，包含一次初次运行及独立复测；不同运行间机器负载不同。
  最新复测为深色 291 ms、浅色 324 ms，首帧约 311 / 347 ms。它是本机隔离 UI 测试，不是所有项目规模的时延保证。
- 真实主窗口和真实 GL 上下文：6 条合成轨迹、6000 个点，使用估算地形；HWND 不变，无根窗口 Hide / Show / WinIdChange。
  首次图形环境初始化仍有成本，未将其移到工作线程或声称无需加载。
- 状态等待有限 UI 动画结束后检查完整窗口截图，深浅设置页、空间三维页正常。
  立即截图会捕获背景颜色动画的中间态，不能据此判断为持久的半套主题。
- 相关回归：92 项（主题、设计令牌、主窗口、空间延后刷新、视图交互、B-scan、参数表单、曲线编辑器）通过；
  另有原生 GL 回归 1 项、输出/传感器/资源树主题回归 5 项、角按钮新增回归 1 项通过，共 99 项。
  独立审查重新运行四项主题测试通过，无剩余复现的 P1/P2。
- Ruff、Python 编译、架构门禁与相关 diff 空白检查通过。未运行本地全量测试。
- 使用独立设置和测试窗口，没有接管用户窗口、修改用户项目、执行采集或增加实机验证。

完整剖析、截图和原生验证脚本保存在当前任务的 `theme-3d-20261102/` 交付目录，没有提交截图或运行日志。
