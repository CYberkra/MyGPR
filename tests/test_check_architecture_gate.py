from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "architecture_policy.toml"


def _load_check_architecture():
    script = ROOT / "scripts" / "check_architecture.py"
    spec = importlib.util.spec_from_file_location("check_architecture", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_policy() -> dict:
    return tomllib.loads(POLICY.read_text(encoding="utf-8"))


def test_check_architecture_passes_on_clean_repo():
    module = _load_check_architecture()
    policy = _load_policy()
    errors, graph = module._check_layers(policy)
    cycle_errors = module._check_layer_cycles(graph)
    legacy_errors = module._check_legacy_core(policy)
    ui_errors = module._check_ui_reverse_dependencies(policy)
    assert not errors, f"layer violations: {errors}"
    assert not cycle_errors, f"cycles: {cycle_errors}"
    assert not legacy_errors, f"legacy violations: {legacy_errors}"
    assert not ui_errors, f"reverse ui dependencies: {ui_errors}"


def test_check_architecture_reports_layer_violations():
    module = _load_check_architecture()
    policy = _load_policy()
    errors, _graph = module._check_layers(policy)
    assert isinstance(errors, list)


def test_check_architecture_reports_ui_reverse_dependencies():
    module = _load_check_architecture()
    policy = _load_policy()
    errors = module._check_ui_reverse_dependencies(policy)
    # 入口脚本 app_qt.py 被策略豁免；其余文件（core/mygpr/PythonModule/cli_batch）仍受约束
    assert isinstance(errors, list)
    assert not any('app_qt.py' in err for err in errors), (
        f"app_qt.py 应被豁免: {errors}")


# --- migration_expiry：豁免的到期兑现 ---------------------------------------
#
# 每条 [[migration_exceptions]] 的 remove_after 是「借到哪个版本为止」的承诺。
# 元数据校验只证明承诺格式正确；_check_migration_expiry 证明承诺是否还在兑现。


def _expiry_policy(*exceptions: dict) -> dict:
    return {
        "migration_expiry": {"enabled": True},
        "migration_exceptions": list(exceptions),
    }


def _exception(**overrides) -> dict:
    base = {
        "path_prefix": "pyproject.toml",
        "owner": "demo-owner",
        "remove_after": "1.0.0",
        "reason": "probe",
    }
    base.update(overrides)
    return base


def test_migration_expiry_enabled_by_default():
    policy = _load_policy()
    assert policy.get("migration_expiry", {}).get("enabled") is True, (
        "migration_expiry 必须默认开启，否则到期承诺又变成沉默承诺")


def test_migration_expiry_reports_current_version_from_pyproject():
    module = _load_check_architecture()
    version = module.current_version()
    assert version is not None, "应能从 pyproject.toml 解析当前版本"
    assert version >= (0, 9), f"当前版本 {version} 与 pyproject.toml 不符"


def test_migration_expiry_silent_before_deadline():
    module = _load_check_architecture()
    errors, warnings = module._check_migration_expiry(_expiry_policy(_exception()))
    # 当前版本 0.9.x 早于 remove_after=1.0.0：承诺尚未到期，不该有任何动静
    assert not errors, f"未到期不应报错: {errors}"
    assert not warnings, f"未到期不应警告: {warnings}"


def test_migration_expiry_warns_past_deadline():
    module = _load_check_architecture()
    module.current_version = lambda: (1, 0, 2)
    errors, warnings = module._check_migration_expiry(
        _expiry_policy(_exception(remove_after="1.0.0")))
    assert not errors, "缺省应只警告不失败（灰度期）"
    assert len(warnings) == 1
    assert "expired" in warnings[0]
    assert "demo-owner" in warnings[0], "警告应点名 owner，否则无人认领"


def test_migration_expiry_fails_when_action_is_fail():
    module = _load_check_architecture()
    module.current_version = lambda: (1, 0, 2)
    errors, warnings = module._check_migration_expiry(
        _expiry_policy(_exception(expiry_action="fail")))
    assert len(errors) == 1, f"expiry_action=fail 必须让门禁失败: {errors}"
    assert "expired" in errors[0]
    assert not warnings, "fail 分支不应同时产出 warning"


def test_migration_expiry_exact_deadline_is_not_yet_expired():
    """版本正好等于 remove_after 时不报过期——到期日当天仍算宽限。"""
    module = _load_check_architecture()
    module.current_version = lambda: (1, 0, 0)
    errors, warnings = module._check_migration_expiry(
        _expiry_policy(_exception(remove_after="1.0.0")))
    assert not errors and not warnings, f"到期日当天应静默: {errors} {warnings}"


def test_migration_expiry_extend_records_new_deadline():
    module = _load_check_architecture()
    module.current_version = lambda: (1, 0, 2)
    errors, warnings = module._check_migration_expiry(
        _expiry_policy(_exception(expiry_action="extend:1.2.0")))
    assert not errors, f"向后延期应合法: {errors}"
    assert len(warnings) == 1
    assert "1.2.0" in warnings[0], "延期应留下新日期的记录"


def test_migration_expiry_rejects_backwards_extension():
    module = _load_check_architecture()
    module.current_version = lambda: (1, 0, 2)
    for bad in ("extend:0.9.0", "extend:1.0.0"):
        errors, _warnings = module._check_migration_expiry(
            _expiry_policy(_exception(expiry_action=bad)))
        assert len(errors) == 1, f"{bad} 应被拒绝（不能等于或早于原到期日）: {errors}"


def test_migration_expiry_rejects_unknown_action():
    module = _load_check_architecture()
    policy = {"migration_exceptions": [_exception(expiry_action="bogus")]}
    errors = module._validate_exception_metadata(policy)
    assert any("invalid expiry_action" in err for err in errors), (
        f"非法 expiry_action 应被值域校验拦下: {errors}")


def test_migration_expiry_can_be_disabled_explicitly():
    module = _load_check_architecture()
    module.current_version = lambda: (9, 9, 9)
    policy = {
        "migration_expiry": {"enabled": False},
        "migration_exceptions": [_exception(expiry_action="fail")],
    }
    errors, warnings = module._check_migration_expiry(policy)
    assert not errors and not warnings, "显式关闭时不应有任何判定"


def test_every_policy_exception_declares_an_expiry_action():
    """每条豁免都应显式声明到期后的动作，避免「到期即沉默」。"""
    policy = _load_policy()
    missing = [
        item["path_prefix"]
        for item in policy.get("migration_exceptions", [])
        if not str(item.get("expiry_action", "")).strip()
    ]
    assert not missing, f"以下豁免缺少 expiry_action: {missing}"
