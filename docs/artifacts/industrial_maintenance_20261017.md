# 工业可靠性维护记录（2026-10-17）

本轮维护分支 codex/maintenance-20261017。保留此前三轮未提交修复、原TODO差异及本地资产；不提交、推送、构建或发布安装包，不改变整体美术设计。

执行顺序为：基线风险 → 各域正式RED与最小修复 → 交叉审查 → 顺序整轮验证 → 工业验收缺口台账。主代理负责规划与审查，执行代理负责源码和验证。详细本地证据在 output/industrial-maintenance-20261017，输出与测试项目不进Git。

存储域本轮确认并修复：stale-lock恢复窗口的双writer竞态；未来schema只读判定前执行恢复；显式只读损坏项目写quarantine；项目创建三阶段失败遗留writer lease；非法对象lock恢复/关闭异常；lock fsync失败遗留活PID锁。修复沿用QC、float32、mmap/分块与原子写约束，公共Backend API和项目数据schema保持现状。

已获得真实独立进程的锁竞争及终止后文件事务恢复证据；未来schema测试检查所有文件字节和mtime不变。存储初始八文件62 passed，交叉审查追加关闭损坏lock回归后直接相关三文件25 passed；最后将失败清理改为成功标志与finally，保留任意异常清理语义，直接四文件29 passed。宽异常数量164低于现有门槛166，未放松门禁。

未来schema关键竞态已补测：首次probe当前schema、writer锁获取前变为未来schema，锁内复检降级只读且不恢复pending事务、不遗留writer lease。最终存储新增回归文件13 passed。

任务域确认并修复提交/关闭的两处竞态和非阻塞关闭后迟到事件误标失败。交互编辑使用隔离候选会话，处理、重放或草稿替换失败时保留数据、头信息、逐道元数据、步骤与撤销/重做；深拷贝嵌套参数，空操作不换会话、不复制整幅当前矩阵。正式初始13 RED，交叉审查后相关七文件49 passed。逐步已保存成果仍是真实资源，后续失败不抹除它们；没有改变QC或显示/科学数据分离。

CLI恢复新增确证误成功：失败摘要列A/B而原配置只剩A，原程序只重跑A就返回0；损坏摘要或自相矛盾stats也可能被当作无失败。结构/完整映射5 RED、stats一致性4 RED后修复，相关三文件40 passed。正常旧摘要无schema/无stats及合法无失败no-op继续兼容。Windows现行构建流程和发布前必需证据文档已同步；只读审查及验证器回归没有发现新的普通打包入口误成功，不重新构建或安装。

上述定向批次存在重叠，不把62/29/49/40简单相加作为总测试数。存储与任务/CLI域已经交叉审查，详细记录在本轮输出目录的storage-audit.md、jobs-audit.md、readiness-audit.md及对应cross-review文件。

## 统一验证与工业验收台账

| 范围 | 状态 | 当前证据与边界 |
| --- | --- | --- |
| 静态与接口检查 | PASS | ruff、架构、schema/API、项目格式、版本一致性、mypy、技术债与复杂度预算通过；门槛不变 |
| 本轮确证故障与定向回归 | PASS | RED→修复→定向与互审完成；隔离项目和测试进程，未操作真实用户数据 |
| 完整源码门禁 | PASS | 本轮完整门禁退出0；1983 passed、6 skipped，pytest 270.90秒；正常执行atexit且无fatal诊断 |
| 实际工作区增量覆盖率 | PASS | core/mygpr变更可执行行462/493，93.71%，高于现行80%门槛；按未提交工作区而非空committed diff计算 |
| 原生Windows入口冒烟/依赖 | PASS | 真实入口/主窗口/后端/七页，8张截图；main正常返回、atexit、退出0且无fatal；pip check通过 |
| 源码一致性与自建测试进程清理 | PASS | 门禁前后652份hash相同、原TODO patch未变；本轮明确标记的测试Python进程残留为0，没有停止用户进程 |
| 局部恢复可靠性 | PARTIAL | 真实测试进程终止、有限事务恢复和确定性I/O故障通过；未覆盖全部断点和文件系统 |
| WIN-FAULT-001商业必需证据 | NOT RUN | artifacts/acceptance/windows/acceptance-evidence.json缺失；真实断电、外盘断连、10GB长跑未验收 |
| CUDA-001商业必需证据 | NOT RUN | artifacts/acceptance/cuda/acceptance-evidence.json缺失；GPU设备/OOM/数值一致性/取消资源验证未验收 |
| HIL-001商业必需证据 | NOT RUN | artifacts/acceptance/hardware/acceptance-evidence.json缺失；真实RTK/IMU/测高/采集时钟未验收 |

源码已记录652个源码/配置文件SHA-256及真实工作区变更行映射；HEAD/分支与原TODO patch核对。原TODO patch含既有尾空白，仍原样保留；排除该文件后的工作区diff-check通过。默认origin/main...HEAD不包含未提交修复，不能据空diff宣称增量覆盖率通过。

全部验证完成后再次逐文件核对652份hash相同，源码未在门禁期间改变。完整结果见output/industrial-maintenance-20261017/summary.json、full-quality-gate.log、diff-coverage-report.json及native-smoke-final/report.json。六项skip为营山外部数据未配置、厂商真实样本未配置，以及已安装PyWavelets时不适用的回退路径；没有降低门槛或把skip作为通过。原生冒烟使用隔离空项目状态，不接管用户窗口，也不替代手工操作、已安装产品或真实数据验收。

本轮完整通过数较前轮1932增加51个参数化用例（存储13、任务/交互23、CLI15），跳过数仍为6。测试进程清理证据见process-cleanup-check.json；仅按本轮输出路径与gate/native显式标记识别自建Python进程，不对无关进程作处置。分支仍为codex/maintenance-20261017，HEAD为e379a5ef602459fef166539f96d3904e7cd9f76a；所有维护修复与小型证据保持未提交，未推送、重打包或发布。

当前证据仍不足以验收真实断电、外盘拔出、网络文件系统锁语义、Linux实机锁分支、多GB性能与内存峰值，以及所有导入/HDF5成果提交中断阶段。固定guard文件只提供内核锁域，不是活锁凭据；旧版writer不识别该guard协议，不承诺混跑旧版时的并发stale恢复安全。它会被当前备份纳入，但不进入报告包；复制后不携带任何内核持锁状态。

交互追加会临时多一幅加载矩阵，重放/reset不额外复制将被替换的current矩阵。Windows目录fsync为空操作；共享POSIX writer在replace成功后目录fsync失败仍可能留下提交不确定性，本轮不声称任意平台/任意I/O故障可回滚。冻结package-check的显式进程退出也不当作自然清理证据，原生GUI正常退出另行检验。

本记录描述源码工作区与局部故障证据，不构成新安装包、物理科学验证、训练就绪或完整工业场景验收声明。
