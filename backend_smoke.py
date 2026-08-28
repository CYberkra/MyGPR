#!/usr/bin/env python3
"""顶层兼容 shim → mygpr.interfaces.cli.backend_smoke。

供 CI（backend-ci.yml）、scripts/run_backend_quality_gate.py 及文档命令
``python backend_smoke.py`` 使用；真实实现在分层后的 mygpr.interfaces.cli 包内。
"""
from __future__ import annotations

from mygpr.interfaces.cli.backend_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
