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
) -> tuple[Path, dict[str, Any]]:
    sdk_root = _ensure_rebot_sdk_in_syspath()
    model_name, data = _load_ros_hardware_config(
        sdk_root,
        hardware_config,
        model,
        channel,
    )
    path = _write_resolved_hardware_config(model_name, data)
    _sync_sdk_robot_model_config(data)
    return path, copy.deepcopy(data)


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
    _add_runtime_config(merged)

    return model_name, merged


def _add_runtime_config(data: dict[str, Any]) -> None:
    arm_joints = _arm_joint_names(data)
    n = len(arm_joints)
    gravity_config = data.get("gravity_compensation", {}) or {}
    control_config = data.get("control", {}) or {}

    data["_runtime"] = {
        "control": {
            "arm_control_mode": _arm_control_mode(data),
            "mit_kp": _control_gain(data, arm_joints, control_config, "mit_kp", "kp"),
            "mit_kd": _control_gain(data, arm_joints, control_config, "mit_kd", "kd"),
        },
        "gravity_compensation": {
            "kp": _gravity_gain(data, arm_joints, gravity_config, "kp"),
            "kd": _gravity_gain(data, arm_joints, gravity_config, "kd"),
            "joint_direction": _runtime_vector(
                gravity_config.get("joint_direction", 1.0),
                n,
                "gravity_compensation.joint_direction",
            ),
            "tau_scale": _runtime_vector(
                gravity_config.get("tau_scale", 1.0),
                n,
                "gravity_compensation.tau_scale",
            ),
        }
    }


def _arm_control_mode(data: dict[str, Any]) -> str:
    mode = str(
        (data.get("control", {}) or {}).get("arm_control_mode", "posvel")
    ).strip().lower()
    if mode == "pos_vel":
        mode = "posvel"
    if mode not in ("posvel", "mit"):
        raise ValueError("control.arm_control_mode must be 'posvel' or 'mit'")
    return mode


def _arm_joint_names(data: dict[str, Any]) -> list[str]:
    joints = data.get("groups", {}).get("arm", {}).get("joints", [])
    if not joints:
        raise ValueError("hardware config must define groups.arm.joints")
    return [str(name) for name in joints]


def _gravity_gain(
    data: dict[str, Any],
    arm_joints: list[str],
    gravity_config: dict[str, Any],
    key: str,
) -> list[float]:
    if key in gravity_config:
        return _runtime_vector(
            gravity_config[key],
            len(arm_joints),
            f"gravity_compensation.{key}",
        )

    joint_map = {str(joint.get("name")): joint for joint in data.get("joints", [])}
    gains = []
    for name in arm_joints:
        joint = joint_map.get(name)
        if joint is None:
            raise ValueError(f"groups.arm references unknown joint {name!r}")
        gains.append(float((joint.get("MIT", {}) or {}).get(key, 0.0)))
    return gains


def _control_gain(
    data: dict[str, Any],
    arm_joints: list[str],
    control_config: dict[str, Any],
    config_key: str,
    mit_key: str,
) -> list[float]:
    if config_key in control_config:
        return _runtime_vector(
            control_config[config_key],
            len(arm_joints),
            f"control.{config_key}",
        )

    joint_map = {str(joint.get("name")): joint for joint in data.get("joints", [])}
    gains = []
    for name in arm_joints:
        joint = joint_map.get(name)
        if joint is None:
            raise ValueError(f"groups.arm references unknown joint {name!r}")
        gains.append(float((joint.get("MIT", {}) or {}).get(mit_key, 0.0)))
    return gains


def _runtime_vector(value: Any, size: int, label: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * size
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a scalar or {size} values")
    values = [float(item) for item in value]
    if len(values) == 1:
        return values * size
    if len(values) != size:
        raise ValueError(f"{label} must be a scalar or {size} values")
    return values


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
