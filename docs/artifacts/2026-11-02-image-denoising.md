# 2026-11-02 高斯平滑 / NLM 算法库接入

## 范围

新增公开处理方法 `gaussian_smoothing`、`non_local_means`，均归入“去噪”。
直接处理带正负的浮点雷达矩阵，保持全部采样、道、物理轴和逐道元数据。
生成新处理成果，沿用处理链、逐步保存和结果 ZIP；原始输入不变。
没有增加依赖、改动采集端或数据协议，没有构建或安装 EXE、提交或推送。
用户已明确本轮先完成两种算法；“真实地形＋多条雷达剖面”留待独立任务。

## 数值与资源约定

- Gaussian 使用 SciPy 分轴高斯滤波，4σ 截断、半采样对称边界；默认时间 σ=0，道间 σ=1。
- NLM 使用均匀波形块的均方距离、指数相似度和自身块，直接加权带符号中心振幅。
  默认块 3采样点×5道、搜索半径 3采样点/5道、h=0.6×全局 RMS；支持绝对振幅单位。
- ndarray 与文件执行共享有重叠边界的内核。每块至多128采样点×256道（另加支持域），
  全局 RMS 分块预扫描；计算和扫描均检查取消，不产生接缝或按块变化的强度尺度。
- 输入 NaN/Inf 明确报错；不转 uint8、不归一化、不截幅、不降采样。
- NLM 使用有限支持正权 `correlate1d`，避免运行和相减在强脉冲后的消减误差；
  用 `sqrt(D)/h` 计算指数，避免极小 h 的平方下溢。
- 新增可选 `file_workspace_bytes`，只供明确实现了有界文件内核的算子声明工作区。
  两算子各声明16MiB；保留文件输入暂存预算以及其他算子的原资源预算。
  8GiB 输入的文件执行 RAM 估计为320MiB，不再误报40/64GiB；该案例为形状估计测试，
  **没有实际处理8GiB文件**。
- 参数表单支持 schema 步长与提示，NLM 奇数块尺寸点击步长为2；步长/提示参与缓存签名。

## 验证

- 先建立失败测试，再实现；复审发现的强脉冲、资源估计、极小h均补充红→绿回归。
- 定向组合 **218 passed in14.33s**：
  `test_image_denoising.py`、`test_native_convergence_baseline.py`、
  `test_native_processing_backend.py`、`test_param_form.py`、
  `test_manual_curve_gain.py`、`test_processing_page_v2.py`。
- 随后新增“文件NLM写出首块后取消，临时工作区清理且不交付半成品”回归，单独1项通过。
- 数值证据包括 SciPy 全幅高斯对照、小矩阵独立标量 NLM、强脉冲前后不同分块高度、
  极小 h、正负反射体合成去噪、坐标/元数据、文件执行、项目成果回读和 ZIP 参数。
- Ruff、compileall、架构门禁通过；未运行本地全量 pytest。
- 独立代码复审最终无剩余P1/P2；复审者独立运行算法35项及既有资源回归3项。

## 本机真实输入与界面证据

只读使用用户已有2026-10-24返程测线H5，`traces` 为1918道×501采样点，
实振幅、时间窗900ns。分块读取完整数据，未抽道；输入文件修改时间/大小保持不变，
处理前后内存原始矩阵 SHA-256 一致。输出均为501×1918、有限float32。
本机第一次调用默认参数计时：Gaussian **0.454s**，NLM **1.297s**。
这些计时包含该次内核调用，不代表所有测区/硬件/搜索半径的性能，也不是去噪解释效果验收。

独立原生Windows窗口验证340px参数区、NLM高级参数展开和浅/深主题控件，
配合B-scan显示全部输出矩阵；没有接管用户软件、覆盖原项目或测试硬件。
截图、验证JSON及临时脚本保存在任务工作目录 `image-denoise-20261102/`，不进Git。
此处是独立控件与后端验证，不能等同安装包或整个主窗口端到端验收。

## 依据与使用说明

- [SciPy Gaussian 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)
- [scikit-image NLM 官方文档](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means)
- [用户说明](../user/reference/图像去噪与平滑.md)
