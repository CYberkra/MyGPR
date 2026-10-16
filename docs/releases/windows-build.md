# Windows 安装包构建

Windows x64 安装包采用 PyInstaller onedir + Inno Setup，内置 Python、Qt 和运行依赖。
目标系统为 Windows 10 1809 及以上 / Windows 11 x64；本轮验证环境为 Windows 11，
未单独验证 Windows 10 或 ARM64。安装到当前用户目录，无需管理员权限。

这是 MyGPR 处理端的独立安装包。CDUT-UavGPR 总平台的子模块、统一启动器及安装器需分别集成，
路径与版本对齐见 [平台对接说明](../developer/CDUT_PLATFORM_INTEGRATION.md#发行与启动入口)。

## 重建

在 Windows x64 的新虚拟环境安装指定发行版 wheel 及构建依赖。
本轮构建解释器为 Python **3.12.12**；3.12 构建需 ≥3.12.12（旧 3.12.0 的冻结 SciPy 导入失败）。
源码开发支持范围仍为 Python 3.12–3.13。

```powershell
py -3.12 -m venv output/windows-build-env
output/windows-build-env/Scripts/python.exe -m pip install "./mygpr-0.11.0-py3-none-any.whl[gui]" -r requirements-windows-build.txt
./packaging/windows/build.ps1 -Python ./output/windows-build-env/Scripts/python.exe -InnoCompiler 'C:/Program Files (x86)/Inno Setup 6/ISCC.exe' -SourceRef v0.11.0
```

工具基线：PyInstaller 6.22.3、hooks-contrib 2026.8、Inno Setup 6.7.3。
冻结阶段使用精简 PATH，拒绝收集构建环境之外的原生 DLL；Qt 运行库固定随 Qt 打包。
先执行冻结程序的后端检查，通过后才压缩安装器。
构建脚本只从 `SourceRef` 的 `git archive` 读取应用源码，不夹带工作区未提交内容。
打包配方独立于应用 tag；不移动既有 tag。结果在 `output/windows-installer-v0.11.0/assets/`。
修改版本时需同步输出目录、源码 ref 和验证脚本中的预期版本。

`packaging/windows/MyGPR.spec` 显式收集动态算法、配置、VERSION 和包元数据；
`MyGPR.iss` 为安装器配方。`launcher.py` 提供 `--package-check <report.json> --hdf5 <file>`，
检查冻结环境版本、处理/自动调参、GIS 驱动、真实 HDF5 矩阵与源文件不变性。
普通 GUI 启动沿用发行版 `app_qt.py`。

## 安装验证

安装 pywinauto 0.6.9 后，在可交互的 Windows 桌面运行：

```powershell
output/windows-build-env/Scripts/python.exe -m pip install pywinauto==0.6.9
output/windows-build-env/Scripts/python.exe packaging/windows/verify_install.py output/windows-installer-v0.11.0/assets/MyGPR-0.11.0-Windows-x64-Setup.exe --expected-version 0.11.0 --hdf5 C:/path/lineData_streaming.h5 --output output/windows-installer-v0.11.0/qa
```

验证使用独立临时安装目录、设置文件和精简 PATH：安装、后端与 HDF5 检查、失败用例返回码、
原生 Windows Qt 七页及整体截图冒烟、重新安装、UIA 设置页版本及正常关闭、卸载和用户文件保留。
截图与完整日志留在 output，不提交 Git。该检查不能代替所有设备、系统版本或科学结论验收。

发布附件使用独立 `windows-build-manifest.json` 和 `WINDOWS-SHA256SUMS.txt`，
记录应用提交、配方哈希、构建环境、验证摘要和安装器 SHA256；不覆盖初次发布的源码/wheel 清单。
安装包包含第三方许可证和完整构建环境包清单（其中含构建/QA 工具，不等于全部都打入应用）。
当前安装器未作代码签名。
