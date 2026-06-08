from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml


def resolve_hardware_config(
    hardware_config: str | None,
    model: str,
    channel: str,
) -> Path:
    sdk_root = _ensure_rebot_sdk_in_syspath()
    model_name, data = _load_ros_hardware_config(
        sdk_root,
        hardware_config,
        model,
        channel,
    )
    path = _write_resolved_hardware_config(model_name, data)
    _sync_sdk_robot_model_config(data)
    return path


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sdk_candidates() -> list[Path]:
    workspace = _workspace_root()
    return [
        workspace / "third_party" / "reBotArm_control_py",
    ]


def _ensure_rebot_sdk_in_syspath() -> Path:
    for root in _sdk_candidates():
        if (root / "reBotArm_control_py").is_dir():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return root
    candidates = "\n".join(f"  - {path}" for path in _sdk_candidates())
    raise FileNotFoundError(
        "Cannot find reBotArm_control_py. Clone it into one of:\n"
        f"{candidates}"
    )


def _default_hardware_config_path() -> Path:
    return (
        _workspace_root()
        / "src"
        / "rebotarm_bringup"
        / "config"
        / "rebotarm_hardware.yaml"
    )


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(override)


def _load_ros_hardware_config(
    sdk_root: Path,
    hardware_config: str | None,
    model: str,
    channel: str,
) -> tuple[str, dict[str, Any]]:
    config_path = (
        Path(hardware_config).expanduser()
        if hardware_config
        else _default_hardware_config_path()
    )
    if not config_path.exists():
        raise FileNotFoundError(f"ROS hardware config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        ros_config = yaml.safe_load(f) or {}

    model_name = (model or ros_config.get("default_model") or "dm").strip().lower()
    models = ros_config.get("models", {})
    if model_name not in models:
        choices = ", ".join(sorted(models))
        raise ValueError(f"unknown hardware model {model_name!r}; choices: {choices}")

    model_config = models[model_name] or {}
    sdk_config = model_config.get("sdk_config")
    if not sdk_config:
        raise ValueError(f"models.{model_name}.sdk_config is required")

    sdk_config_path = Path(str(sdk_config)).expanduser()
    if not sdk_config_path.is_absolute():
        sdk_config_path = sdk_root / "config" / sdk_config_path
    if not sdk_config_path.exists():
        raise FileNotFoundError(f"SDK hardware config not found: {sdk_config_path}")

    with open(sdk_config_path, "r", encoding="utf-8") as f:
        merged = yaml.safe_load(f) or {}

    merged = _deep_merge(merged, model_config.get("overrides", {}) or {})
    if channel:
        merged["channel"] = channel

    return model_name, merged


def _write_resolved_hardware_config(model: str, data: dict[str, Any]) -> Path:
    tmp_dir = Path("/tmp") / "rebotarm_ros2"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{model}_hardware.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return tmp_path


def _sync_sdk_robot_model_config(data: dict[str, Any]) -> None:
    import reBotArm_control_py.kinematics.robot_model as robot_model
    import reBotArm_control_py.dynamics.robot_model as dynamics_model

    robot_model._hw_cfg_cache = copy.deepcopy(data)
    dynamics_model._CACHED_MODEL = None
