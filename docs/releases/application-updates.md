# Windows 应用内更新发布

安装版在“设置 → 关于”检查更新。自动检查最多每天一次；渠道切换保存后重启生效。下载、安装、稍后、跳过与取消由 WinSparkle 对话框处理。源码运行只提供发布页入口。存在活动任务、未保存标注或设置保存失败时拒绝安装。

Ed25519 更新签名是必需的，无需购买证书。Windows Authenticode 证书是可选项；没有它仍可制作带验签更新的安装包，但 Windows 可能显示未知发布者提示。两种签名解决不同问题，不能互相替代。

## 一次性生成发布密钥

在发布者的 Windows 账户执行，私钥必须位于仓库之外。命令只输出公钥；私钥文件权限限制为当前账户和 SYSTEM，且拒绝覆盖已有密钥。

```powershell
.\.venv\Scripts\python.exe -m scripts.publish_windows_update --generate-key --private-key "$env:USERPROFILE/.mygpr-publisher/windows-ed25519.key"
```

将输出的公钥保存到 `windows-ed25519.pub`，并离线备份私钥。私钥不得进入 Git、发布附件或普通 CI 日志。首轮实现不将私钥上传 GitHub。密钥遗失时，已安装版本无法信任新密钥签名的更新，需人工安装新的信任根版本。

正式公钥现已固定在 `config/windows_updates.json`。构建器默认使用该公钥；`-UpdatePublicKeyFile` 是显式覆盖，空文件会被拒绝。私钥仍只保存在发布者账户的仓库外目录。创建文件与限制权限不等于已经完成离线备份。

## 构建与验收

先提交源码，并把同一提交构建的 wheel 安装到隔离的构建环境。以下 `SourceRef` 使用已经提交的不可变 SHA。完整构建基线见 [Windows 构建](windows-build.md)。

```powershell
./packaging/windows/build.ps1 -Python ./output/windows-build-env/Scripts/python.exe -InnoCompiler 'C:/Program Files (x86)/Inno Setup 6/ISCC.exe' -SourceRef <commit-sha> -OutputRoot output/windows-updater-build -UpdatePublicKeyFile "$env:USERPROFILE/.mygpr-publisher/windows-ed25519.pub"
```

可选 `-CertificateThumbprint <40位证书指纹>`：使用 Windows SDK 的 signtool 签署程序、卸载器和安装器，并验证 Authenticode。没有证书时省略；不降低更新载荷验签要求。`-PreviewFeed <独立HTTPS订阅地址>` 才启用预览渠道。

构建产物记录内置公钥、源码提交、配方哈希和安装包哈希。官方 WinSparkle 归档固定为 0.9.4，SHA-256 为 `6037df37fc263bd1650a1c4949681a9d40ffe991d01f35892a406cb5d103c976`；构建阶段下载并核对，应用运行时不下载 DLL。

首轮升级还需在干净 Windows 账户运行完整安装验收。在原 `verify_install.py` 命令后增加 `--previous-installer <旧安装器.exe>`，先安装旧版，再覆盖新版本并检查设置、项目文件和原始 HDF5 不变。该工具会拒绝覆盖用户已有的注册安装。

安装器不强制终止软件。新安装版持有互斥锁，程序退出前不能覆盖安装。旧依赖只按旧文件清单与内容哈希删除；修改过的文件、未知文件、越界路径和 junction 均保留。早期没有清单的版本不执行清理。更新取消或失败时不承诺文件系统回滚，尤其不能把断电恢复视为已验证。

## 签名并准备订阅

使用最终安装包和同次构建的 `build-provenance.json`。脚本先验证安装包哈希及内置公钥与私钥相符，再签名、验签并原子写出文件。

```powershell
.\.venv\Scripts\python.exe -m scripts.publish_windows_update --private-key "$env:USERPROFILE/.mygpr-publisher/windows-ed25519.key" --installer output/windows-updater-build/assets/MyGPR-0.11.4-Windows-x64-Setup.exe --build-manifest output/windows-updater-build/build-provenance.json --version 0.11.4 --url https://github.com/CYberkra/MyGPR/releases/download/v0.11.4/MyGPR-0.11.4-Windows-x64-Setup.exe --notes-url https://github.com/CYberkra/MyGPR/releases/tag/v0.11.4 --channel stable --output output/windows-updater-feed
```

命令不发布任何内容。版本及 URL 必须换成实际新发行版，不能移动已有 tag 或替换已发布安装包。预览版本支持 `aN`、`bN`、`rcN`，必须用 `--channel preview` 和独立 feed。Windows 数值构建号保证同一版本的 alpha < beta < rc < final。

先上传安装包、说明、校验和、构建/验收证据，最后让 `appcast.xml` 可访问。稳定订阅为 `https://github.com/CYberkra/MyGPR/releases/latest/download/appcast.xml`。发布流程须验证该地址实际指向目标正式版；发布预览版时不改变 stable latest。启用不可变 Releases 后也必须在最终发布前备齐附件。

现有未带更新器的版本需要用户手动安装一次。首次线上验收还需覆盖：离线、下载中断、篡改载荷、跳过、取消、活动任务和已安装版本的真实升级。常规 Qt CI、真实签名用例和小型安装器夹具各有边界，不能代替这次验收。

## GitHub 完整安装与更新验收

`MyGPR Windows packages` 从不可变提交构建两个完整应用：正式版本，以及仅供验收的 `a1` 旧版本。后者使用独立 localhost HTTPS 订阅，不得发布给用户。GitHub 使用隔离的 Python 3.13 环境；自行采用 3.12 构建时仍要求补丁版至少 3.12.12。

发布者下载 `windows-package-release` 附件，在本机用仓库外私钥签名。只将公开的安装包签名传给后续验收，私钥不离开本机：

```powershell
gh workflow run focused-regressions.yml --repo CYberkra/MyGPR --ref <feature-branch> -f suite=update_installation -f package_run=<successful-package-run-id> -f installer_signature=<public-ed25519-signature>
```

该入口只在一次性 Windows runner 上运行。它核对来源分支、构建清单、签名及程序源码是否改变。它验证完整 v0.11.0 安装器覆盖升级，以及完整旧版应用经 WinSparkle 的真实 HTTPS 发现、下载、签名验证、退出、安装、重启。还验证离线、已是最新、稍后、跳过状态保存、下载取消、中断和无效签名。程序重启关闭时必须返回退出码 0。

测试创建小型合成 HDF5、真实 MyGPR 项目和处理成果，检查更新后矩阵值及文件哈希不变，设置仍保留。HTTPS 测试证书只进入一次性 runner 的证书存储并在收尾清理；不会修改发布者电脑的证书设置。它证明隔离环境的完整流程，不证明断电恢复、所有 Windows 版本或正式公网订阅已经发布。

参考：[WinSparkle 接入](https://winsparkle.org/guides/getting-started/)、[更新发布](https://winsparkle.org/guides/publishing-updates/)、[回调顺序](https://winsparkle.org/c-api/callbacks/)、[Inno 签名](https://jrsoftware.org/ishelp/topic_setup_signtool.htm)、[GitHub 不可变 Releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)。
