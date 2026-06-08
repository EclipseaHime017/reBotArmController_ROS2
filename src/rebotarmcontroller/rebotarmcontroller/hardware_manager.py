from __future__ import annotations

import time

import numpy as np

from .conversions import fk_to_pose
from .hardware_config import resolve_hardware_config

_GC_VEL_THRESHOLD = 0.04
_GC_W_VEL_THRESHOLD = 0.08
_GC_KP = 7.0
_GC_KD = 0.8
_GRIPPER_GOAL_TOLERANCE_RAD = 0.12
_GRIPPER_CLOSED_POSITION = 0.0


class HardwareManager:
    """ROS-facing adapter for the new grouped reBotArm SDK."""

    def __init__(
        self,
        hardware_config: str | None = None,
        model: str = "",
        channel: str = "",
    ) -> None:
        hardware_config_path = resolve_hardware_config(
            hardware_config,
            model,
            channel,
        )

        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.controllers import RebotArmEndPose
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import (
            get_end_effector_frame_id,
            load_robot_model,
        )
        import pinocchio as pin

        self._robot = RebotArm(hw_yaml=str(hardware_config_path))
        self._arm_group = self._robot.groups.get("arm")
        if self._arm_group is None:
            raise ValueError("hardware config must define groups.arm")
        self._gripper_group = self._robot.groups.get("gripper")
        self._robot_get_state = self._robot.get_state
        self._robot.get_state = self._get_arm_state
        self._endpos_ctrl = RebotArmEndPose(
            self._robot,
            arm_control_mode="posvel",
        )

        self._gripper_name = (
            self._gripper_group.joint_names[0]
            if self.has_gripper and self._gripper_group.joint_names
            else ""
        )
        self._gripper_target_position: float | None = None

        self._gc_model = load_robot_model()
        self._gc_data = self._gc_model.createData()
        self._gc_ee_frame_id = get_end_effector_frame_id(self._gc_model)
        self._gc_compute_generalized_gravity = compute_generalized_gravity
        self._gc_pin = pin

        self._connected = False
        self._enabled = False
        self._control_output_enabled = False
        self._state_machine = "IDLE"
        self._error_codes: list[str] = []
        self._gravity_comp_active = False
        self._gravity_comp_q_target: np.ndarray | None = None
        self._gravity_comp_integral: np.ndarray | None = None
        self._gravity_comp_lock_counter = 0
        self._gravity_comp_q_last: np.ndarray | None = None

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def endpos_ctrl(self):
        return self._endpos_ctrl

    @property
    def joint_names(self) -> list[str]:
        return list(self._arm_group.joint_names)

    @property
    def mode(self) -> str:
        return str(self._arm_group.mode)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def control_loop_active(self) -> bool:
        return bool(self._robot.control_loop_active)

    @property
    def has_gripper(self) -> bool:
        return bool(self._robot.has_gripper)

    @property
    def state_machine(self) -> str:
        return self._state_machine

    @property
    def error_codes(self) -> list[str]:
        return list(self._error_codes)

    def set_state_machine(self, state: str) -> None:
        if state not in ("IDLE", "TRAJ_RUNNING", "LOWLEVEL_STREAMING", "GRAVITY_COMP"):
            raise ValueError(f"unsupported state machine value: {state}")
        self._state_machine = state

    # ------------------------------------------------------------------
    # arm
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            return
        try:
            self._robot.connect()
            self._configure_groups_for_endpos()
            if self.has_gripper:
                self._gripper_target_position = self.get_gripper_state()[0]
            self.hold_current_position()
            self._control_output_enabled = True
            self._robot.start_control_loop(self._endpos_loop_cb)
            self._endpos_ctrl._running = True
            self._connected = True
            self._enabled = True
        except Exception:
            self._control_output_enabled = False
            self._endpos_ctrl._running = False
            try:
                self._robot.stop_control_loop()
                self._robot.disconnect()
            finally:
                self._connected = False
                self._enabled = False
            raise

    def shutdown(
        self,
        safe_home: bool = True,
        disable_after_safe_home: bool = True,
    ) -> None:
        if not self._connected:
            return
        try:
            self.stop_gravity_compensation()

            if safe_home:
                self.safe_home()

            self.stop_motion()
            self._robot.stop_control_loop()

            if disable_after_safe_home:
                self.disable()

            self._robot.disconnect()
        finally:
            self._connected = False
            self._enabled = False
            self._control_output_enabled = False
            self.set_state_machine("IDLE")

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._get_arm_state()

    def _get_arm_state(
        self,
        request_feedback: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos, vel, torq = self._robot_get_state(request_feedback=request_feedback)
        n = len(self.joint_names)
        return pos[:n], vel[:n], torq[:n]

    def get_joint_positions(self, request: bool = False) -> np.ndarray:
        return self._arm_group.get_positions(request_feedback=request)

    def get_joint_velocities(self, request: bool = False) -> np.ndarray:
        return self._arm_group.get_velocities(request_feedback=request)

    def hold_current_position(self) -> np.ndarray:
        current = self.get_joint_positions(request=True).copy()
        self._endpos_ctrl._q_target[:] = current
        self._endpos_ctrl._qd_target[:] = 0.0
        return current

    def set_joint_position_target(self, positions) -> None:
        target = np.asarray(positions, dtype=np.float64).reshape(-1)
        if len(target) != len(self.joint_names):
            raise ValueError(
                f"expected {len(self.joint_names)} joint targets, got {len(target)}"
            )
        self.stop_motion()
        self._endpos_ctrl._q_target[:] = target
        self._endpos_ctrl._qd_target[:] = 0.0

    def start_endpos_control(self) -> None:
        if self._gravity_comp_active:
            raise RuntimeError("stop gravity compensation before starting endpos control")

        if self.control_loop_active:
            self._control_output_enabled = False
            self._retry_hardware_call(self._arm_group.enable, "arm enable")
            if self.has_gripper:
                self._retry_hardware_call(self._gripper_group.enable, "gripper enable")
            self.hold_current_position()
            self._endpos_ctrl._running = True
            self._control_output_enabled = True
            self._enabled = True
            self.set_state_machine("IDLE")
            return

        self._robot.stop_control_loop()
        self._configure_groups_for_endpos()
        self.hold_current_position()
        self._control_output_enabled = True
        self._robot.start_control_loop(self._endpos_loop_cb)
        self._endpos_ctrl._running = True
        self._enabled = True
        self.set_state_machine("IDLE")

    def enable(self) -> None:
        self.start_endpos_control()

    def disable(self) -> None:
        if self._gravity_comp_active:
            raise RuntimeError("stop gravity compensation before disable")
        self._control_output_enabled = False
        self.stop_motion()
        self._endpos_ctrl._running = False
        self._robot.disable_all()
        self._enabled = False
        self.set_state_machine("IDLE")

    def safe_home(self) -> None:
        self.stop_gravity_compensation()
        self.start_endpos_control()
        if self.has_gripper:
            self.set_gripper_position(_GRIPPER_CLOSED_POSITION)
        self._endpos_ctrl.safe_home()

    def set_mode(self, mode: str) -> bool:
        mode = mode.strip().lower()
        if mode not in ("mit", "pos_vel", "vel"):
            raise ValueError(f"unsupported mode: {mode}")
        self.stop_gravity_compensation()
        self.stop_motion()
        self._robot.stop_control_loop()
        self._endpos_ctrl._running = False

        if mode == "mit":
            ok = self._arm_group.mode_mit()
        elif mode == "pos_vel":
            ok = self._arm_group.mode_pos_vel()
            if self._enabled:
                self._start_pos_vel_hold()
        else:
            ok = self._arm_group.mode_vel()
        self.set_state_machine("IDLE")
        return bool(ok)

    def set_zero(self, joint_name: str = "") -> bool:
        self.stop_motion()
        self._robot.stop_control_loop()
        self._endpos_ctrl._running = False
        if joint_name:
            if joint_name not in self._robot._motor_map:
                raise KeyError(f"unknown joint: {joint_name}")
            self._set_zero_single(joint_name)
        else:
            self._robot.set_zero()
        self._enabled = False
        self.set_state_machine("IDLE")
        return True

    def send_joint_mit_cmd(
        self,
        joint_name: str,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        tau: float,
    ) -> None:
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("mit")
        q = self._arm_group.get_positions(request_feedback=True)
        target_pos = np.array(q, dtype=np.float64, copy=True)
        target_vel = np.zeros(len(self.joint_names), dtype=np.float64)
        target_tau = np.zeros(len(self.joint_names), dtype=np.float64)
        target_kp = np.array(
            getattr(self._arm_group, "_mit_kp"),
            dtype=np.float64,
            copy=True,
        )
        target_kd = np.array(
            getattr(self._arm_group, "_mit_kd"),
            dtype=np.float64,
            copy=True,
        )
        target_pos[index] = float(pos)
        target_vel[index] = float(vel)
        target_kp[index] = float(kp)
        target_kd[index] = float(kd)
        target_tau[index] = float(tau)
        self._arm_group.send_mit(
            target_pos,
            vel=target_vel,
            kp=target_kp,
            kd=target_kd,
            tau=target_tau,
        )
        self.set_state_machine("LOWLEVEL_STREAMING")

    def send_joint_pos_vel_cmd(
        self,
        joint_name: str,
        pos: float,
        vlim: float,
    ) -> None:
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("pos_vel")
        q = self._arm_group.get_positions(request_feedback=True)
        target_pos = np.array(q, dtype=np.float64, copy=True)
        target_vlim = np.array(
            getattr(self._arm_group, "_pv_vlim"),
            dtype=np.float64,
            copy=True,
        )
        target_pos[index] = float(pos)
        target_vlim[index] = float(vlim)
        self._arm_group.send_pos_vel(target_pos, vlim=target_vlim)
        self.set_state_machine("LOWLEVEL_STREAMING")

    def send_joint_vel_cmd(self, joint_name: str, vel: float) -> None:
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("vel")
        target_vel = np.zeros(len(self.joint_names), dtype=np.float64)
        target_vel[index] = float(vel)
        self._arm_group.send_vel(target_vel)
        self.set_state_machine("LOWLEVEL_STREAMING")

    def current_pose(self):
        from reBotArm_control_py.kinematics import compute_fk, pad_q_for_model

        q, _, _ = self.get_joint_state()
        q_padded = pad_q_for_model(self._gc_model, q, len(self.joint_names))
        position, rotation, _ = compute_fk(self._gc_model, q_padded)
        return fk_to_pose(position, rotation)

    def get_joint_status_codes(self) -> list[int]:
        codes: list[int] = []
        for name in self.joint_names:
            try:
                st = self._robot._motor_map[name].get_state()
                codes.append(int(st.status_code if st is not None else 0))
            except Exception:
                codes.append(0)
        return codes

    # ------------------------------------------------------------------
    # gravity compensation
    # ------------------------------------------------------------------

    def start_gravity_compensation(self) -> None:
        self.stop_gravity_compensation()
        if not self._enabled:
            self._arm_group.enable()
            if self.has_gripper:
                self._gripper_group.enable()
            self._enabled = True
        self.stop_motion()
        self._robot.stop_control_loop()
        self._endpos_ctrl._running = False

        self._gravity_comp_q_target = self._arm_group.get_positions(
            request_feedback=True,
        ).copy()
        self._gravity_comp_q_last = self._gravity_comp_q_target.copy()
        self._arm_group.mode_mit(
            kp=np.full(len(self.joint_names), _GC_KP, dtype=np.float64),
            kd=np.full(len(self.joint_names), _GC_KD, dtype=np.float64),
        )
        self._gravity_comp_integral = np.zeros_like(self._gravity_comp_q_target)
        self._gravity_comp_lock_counter = 0
        self._gravity_comp_active = True
        arm_rate = float(getattr(self._robot, "_rate", 500.0))
        self._gravity_comp_tick(self._robot, 1.0 / arm_rate)
        self._robot.start_control_loop(self._gravity_comp_tick, rate=arm_rate)
        self.set_state_machine("GRAVITY_COMP")

    def stop_gravity_compensation(self) -> None:
        if not self._gravity_comp_active:
            return
        hold_target = (
            self._gravity_comp_q_last.copy()
            if self._gravity_comp_q_last is not None
            else None
        )
        self._robot.stop_control_loop()
        self._gravity_comp_active = False
        self._gravity_comp_q_target = None
        self._gravity_comp_integral = None
        self._gravity_comp_lock_counter = 0
        self._gravity_comp_q_last = None
        if self._enabled:
            self._arm_group.mode_pos_vel()
            self._start_pos_vel_hold(target=hold_target)
        self.set_state_machine("IDLE")

    def gravity_compensation_active(self) -> bool:
        return self._gravity_comp_active

    def gravity_compensation_target(self) -> np.ndarray | None:
        if self._gravity_comp_q_target is None:
            return None
        return self._gravity_comp_q_target.copy()

    @staticmethod
    def _angles_near_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        delta = values - reference
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        return reference + delta

    def _read_gravity_comp_positions(
        self,
        *,
        request: bool = False,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        q = self._arm_group.get_positions(request_feedback=request)
        ref = reference if reference is not None else self._gravity_comp_q_last
        if ref is not None:
            q = self._angles_near_reference(q, ref)
        self._gravity_comp_q_last = np.array(q, dtype=np.float64, copy=True)
        return self._gravity_comp_q_last.copy()

    def _gravity_comp_tick(self, _robot, dt: float) -> None:
        del dt
        if not self._gravity_comp_active or self._gravity_comp_q_target is None:
            return

        from reBotArm_control_py.kinematics import pad_q_for_model

        q = self._read_gravity_comp_positions()
        qd = self._arm_group.get_velocities(request_feedback=False)
        q_model = pad_q_for_model(self._gc_model, q, len(self.joint_names))
        qd_model = np.zeros(self._gc_model.nv, dtype=np.float64)
        qd_model[: min(len(qd), len(self.joint_names), self._gc_model.nv)] = qd[
            : min(len(qd), len(self.joint_names), self._gc_model.nv)
        ]
        tau_g = self._gc_compute_generalized_gravity(
            self._gc_model,
            q_model,
            self._gc_data,
        )[: len(self.joint_names)]

        q_error = self._gravity_comp_q_target - q
        if self._gravity_comp_integral is None:
            self._gravity_comp_integral = np.zeros_like(q)
        self._gravity_comp_integral += q_error * 1.0
        np.clip(self._gravity_comp_integral, -0.5, 0.5, out=self._gravity_comp_integral)

        self._gc_pin.computeJointJacobians(self._gc_model, self._gc_data, q_model)
        self._gc_pin.updateFramePlacements(self._gc_model, self._gc_data)
        jacobian = self._gc_pin.getFrameJacobian(
            self._gc_model,
            self._gc_data,
            self._gc_ee_frame_id,
            self._gc_pin.ReferenceFrame.WORLD,
        )
        spatial_velocity = jacobian @ qd_model
        linear_speed = float(np.linalg.norm(spatial_velocity[:3]))
        angular_speed = float(np.linalg.norm(spatial_velocity[3:]))

        if linear_speed > _GC_VEL_THRESHOLD or angular_speed > _GC_W_VEL_THRESHOLD:
            self._gravity_comp_q_target = q.copy()
            self._gravity_comp_lock_counter = 0
            self._gravity_comp_integral *= 0.9
        else:
            self._gravity_comp_lock_counter += 1

        self._arm_group.send_mit(
            self._gravity_comp_q_target,
            vel=np.zeros(len(self.joint_names)),
            kp=np.full(len(self.joint_names), _GC_KP),
            kd=np.full(len(self.joint_names), _GC_KD),
            tau=tau_g + self._gravity_comp_integral,
        )

    # ------------------------------------------------------------------
    # gripper
    # ------------------------------------------------------------------

    def set_gripper_target(self, position: float) -> None:
        self._begin_gripper_command(allow_endpos=True)
        if self.control_loop_active and self._endpos_ctrl._running:
            self._gripper_group.mode_mit()
            self._endpos_ctrl.set_gripper_target(float(position))
        else:
            self._gripper_group.mode_pos_vel()
            self._gripper_group.send_pos_vel(
                np.array([float(position)], dtype=np.float64),
                vlim=np.array(
                    getattr(self._gripper_group, "_pv_vlim"),
                    dtype=np.float64,
                    copy=True,
                ),
            )
        self._gripper_target_position = float(position)

    def wait_gripper_target(self, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.gripper_reached_target():
                return True
            time.sleep(0.02)
        return False

    def set_gripper_position(
        self,
        position: float,
        timeout: float = 3.0,
    ) -> tuple[bool, float]:
        self.set_gripper_target(position)
        reached = self.wait_gripper_target(timeout)
        return reached, self.get_gripper_state()[0]

    def get_gripper_state(self) -> tuple[float, float, float, int]:
        if not self.has_gripper or not self._gripper_name:
            return 0.0, 0.0, 0.0, 0
        pos = float(self._gripper_group.get_positions()[0])
        vel = float(self._gripper_group.get_velocities(request_feedback=False)[0])
        status = 0
        torque = 0.0
        try:
            st = self._robot._motor_map[self._gripper_name].get_state()
            if st is not None:
                torque = float(st.torq)
                status = int(st.status_code)
        except Exception:
            status = 0
        return float(pos), float(vel), float(torque), status

    def gripper_reached_target(self) -> bool:
        if self._gripper_target_position is None:
            return True
        pos = self.get_gripper_state()[0]
        return abs(pos - self._gripper_target_position) < _GRIPPER_GOAL_TOLERANCE_RAD

    def send_gripper_mit_cmd(
        self,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        tau: float,
    ) -> None:
        self._begin_gripper_command()
        self._begin_gripper_lowlevel("mit")
        self._gripper_group.send_mit(
            np.array([float(pos)], dtype=np.float64),
            vel=np.array([float(vel)], dtype=np.float64),
            kp=np.array([float(kp)], dtype=np.float64),
            kd=np.array([float(kd)], dtype=np.float64),
            tau=np.array([float(tau)], dtype=np.float64),
        )
        self._gripper_target_position = None

    def send_gripper_pos_vel_cmd(self, pos: float, vlim: float) -> None:
        self._begin_gripper_command()
        self._begin_gripper_lowlevel("pos_vel")
        self._gripper_group.send_pos_vel(
            np.array([float(pos)], dtype=np.float64),
            vlim=np.array([float(vlim)], dtype=np.float64),
        )
        self._gripper_target_position = None

    def send_gripper_vel_cmd(self, vel: float) -> None:
        self._begin_gripper_command()
        self._begin_gripper_lowlevel("vel")
        self._gripper_group.send_vel(np.array([float(vel)], dtype=np.float64))
        self._gripper_target_position = None

    def _begin_gripper_command(self, *, allow_endpos: bool = False) -> None:
        if not self._enabled:
            raise RuntimeError("rejecting gripper command while arm is disabled")
        if self._gravity_comp_active or self.state_machine == "GRAVITY_COMP":
            raise RuntimeError("rejecting gripper command during gravity compensation")
        if self.state_machine == "TRAJ_RUNNING":
            raise RuntimeError("rejecting gripper command while trajectory is running")
        if not self.has_gripper or not self._gripper_name:
            raise RuntimeError("gripper is not initialized")
        if not allow_endpos and self.control_loop_active:
            self.stop_motion()
            self._robot.stop_control_loop()
            self._endpos_ctrl._running = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _begin_lowlevel_streaming(self, required_mode: str) -> None:
        if not self._enabled:
            raise RuntimeError("rejecting low-level command while arm is disabled")
        if self._gravity_comp_active or self.state_machine == "GRAVITY_COMP":
            raise RuntimeError("rejecting low-level command during gravity compensation")
        if self.state_machine == "TRAJ_RUNNING":
            self.stop_motion()
        self._robot.stop_control_loop()
        self._endpos_ctrl._running = False

        if required_mode == self.mode:
            self.set_state_machine("LOWLEVEL_STREAMING")
            return
        if required_mode == "mit":
            ok = self._arm_group.mode_mit()
        elif required_mode == "pos_vel":
            ok = self._arm_group.mode_pos_vel()
        elif required_mode == "vel":
            ok = self._arm_group.mode_vel()
        else:
            raise ValueError(f"unsupported low-level mode: {required_mode}")
        if not ok:
            raise RuntimeError(f"not all arm joints entered {required_mode} mode")
        self.set_state_machine("LOWLEVEL_STREAMING")

    def _begin_gripper_lowlevel(self, required_mode: str) -> None:
        if required_mode == "mit":
            ok = self._gripper_group.mode_mit()
        elif required_mode == "pos_vel":
            ok = self._gripper_group.mode_pos_vel()
        elif required_mode == "vel":
            ok = self._gripper_group.mode_vel()
        else:
            raise ValueError(f"unsupported gripper mode: {required_mode}")
        if not ok:
            raise RuntimeError(f"gripper did not enter {required_mode} mode")
        self.set_state_machine("LOWLEVEL_STREAMING")

    def _start_pos_vel_hold(self, target: np.ndarray | None = None) -> None:
        if self.control_loop_active:
            return
        self._configure_groups_for_endpos()
        if target is None:
            self.hold_current_position()
        else:
            self._endpos_ctrl._q_target[:] = np.array(target, dtype=np.float64)
            self._endpos_ctrl._qd_target[:] = 0.0
        self._endpos_ctrl._running = True
        self._control_output_enabled = True
        self._robot.start_control_loop(self._endpos_loop_cb)

    def _configure_groups_for_endpos(self) -> None:
        self._retry_hardware_call(self._arm_group.mode_pos_vel, "arm pos_vel mode")
        self._retry_hardware_call(self._arm_group.enable, "arm enable")
        if self.has_gripper:
            self._retry_hardware_call(self._gripper_group.mode_mit, "gripper mit mode")
            self._retry_hardware_call(self._gripper_group.enable, "gripper enable")

    @staticmethod
    def _retry_hardware_call(fn, label: str, attempts: int = 3):
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                time.sleep(0.2 * attempt)
        raise RuntimeError(f"{label} failed after {attempts} attempts: {last_exc}") from last_exc

    def stop_motion(self) -> None:
        self._endpos_ctrl._stop_send.set()
        if self._endpos_ctrl._send_thread is not None:
            self._endpos_ctrl._send_thread.join(timeout=5.0)
        self._endpos_ctrl._moving = False
        self._endpos_ctrl._stop_send.clear()

    def motion_active(self) -> bool:
        return bool(self._endpos_ctrl._moving)

    def _endpos_loop_cb(self, robot, dt: float) -> None:
        if not self._control_output_enabled:
            return
        self._endpos_ctrl._loop_cb(robot, dt)

    def _joint_index(self, joint_name: str) -> int:
        try:
            return self.joint_names.index(joint_name)
        except ValueError as exc:
            raise KeyError(f"unknown joint: {joint_name}") from exc

    def _set_zero_single(self, joint_name: str) -> None:
        self._robot.disable_all()
        time.sleep(0.3)
        motor = self._robot._motor_map[joint_name]
        ctrl = None
        for joint in self._robot._all_joints:
            if joint.name == joint_name:
                ctrl = self._robot._ctrl_map[str(joint.vendor)]
                break
        if ctrl is None:
            raise KeyError(f"unknown joint: {joint_name}")
        for _ in range(200):
            try:
                motor.request_feedback()
                ctrl.poll_feedback_once()
            except Exception:
                pass
            st = motor.get_state()
            if st is not None and st.status_code == 0:
                break
            time.sleep(0.05)
        motor.set_zero_position()
