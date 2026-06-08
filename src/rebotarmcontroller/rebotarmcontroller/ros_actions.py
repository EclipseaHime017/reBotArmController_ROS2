from __future__ import annotations

import time

from control_msgs.action import FollowJointTrajectory, GripperCommand
import numpy as np
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rebotarm_msgs.action import MoveToPose

from .conversions import pose_to_xyz_rpy


class ArmActions:
    def __init__(self, node, hardware, namespace: str) -> None:
        self._node = node
        self._hardware = hardware
        self._namespace = namespace
        self._move_to_pose_server = ActionServer(
            node,
            MoveToPose,
            f"/{namespace}/move_to_pose",
            execute_callback=self.execute_move_to_pose,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_move_to_pose,
            callback_group=node.reentrant_group,
        )
        self._follow_joint_trajectory_server = ActionServer(
            node,
            FollowJointTrajectory,
            f"/{namespace}/follow_joint_trajectory",
            execute_callback=self.execute_follow_joint_trajectory,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_follow_joint_trajectory,
            callback_group=node.reentrant_group,
        )
        self._gripper_command_server = ActionServer(
            node,
            GripperCommand,
            f"/{namespace}/gripper/command",
            execute_callback=self.execute_gripper_command,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_gripper_command,
            callback_group=node.reentrant_group,
        )

    def goal_callback(self, _goal_request):
        return GoalResponse.ACCEPT

    def cancel_move_to_pose(self, _goal_handle):
        self._hardware.stop_motion()
        return CancelResponse.ACCEPT

    def cancel_follow_joint_trajectory(self, _goal_handle):
        return CancelResponse.ACCEPT

    def cancel_gripper_command(self, _goal_handle):
        return CancelResponse.ACCEPT

    def execute_move_to_pose(self, goal_handle):
        goal = goal_handle.request
        result = MoveToPose.Result()

        try:
            self._hardware.start_endpos_control()
            self._hardware.set_state_machine("TRAJ_RUNNING")
            self._node.publish_arm_status()
            x, y, z, roll, pitch, yaw = pose_to_xyz_rpy(goal.target_pose)
            ok = self._hardware.endpos_ctrl.move_to_traj(
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
                float(goal.duration),
            )
        except Exception as exc:
            self._hardware.hold_current_position()
            self._hardware.set_state_machine("IDLE")
            self._node.publish_arm_status()
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            result.final_pose = self._hardware.current_pose()
            return result

        if not ok:
            self._hardware.set_state_machine("IDLE")
            self._node.publish_arm_status()
            goal_handle.abort()
            result.success = False
            result.message = "trajectory planning failed"
            result.final_pose = self._hardware.current_pose()
            return result

        deadline = time.monotonic() + max(float(goal.duration), 0.0) + 2.0
        while self._hardware.motion_active():
            if goal_handle.is_cancel_requested:
                self._hardware.stop_motion()
                self._hardware.hold_current_position()
                self._hardware.set_state_machine("IDLE")
                self._node.publish_arm_status()
                goal_handle.canceled()
                result.success = False
                result.message = "move_to_pose canceled"
                result.final_pose = self._hardware.current_pose()
                return result
            if time.monotonic() > deadline:
                self._hardware.stop_motion()
                self._hardware.hold_current_position()
                self._hardware.set_state_machine("IDLE")
                self._node.publish_arm_status()
                goal_handle.abort()
                result.success = False
                result.message = "move_to_pose timeout"
                result.final_pose = self._hardware.current_pose()
                return result
            time.sleep(0.02)

        positions = self._hardware.get_joint_positions()
        velocities = self._hardware.get_joint_velocities()
        result.success = True
        result.message = (
            "move_to_traj accepted "
            f"positions={[float(v) for v in positions]} "
            f"velocities={[float(v) for v in velocities]}"
        )
        result.final_pose = self._hardware.current_pose()
        self._hardware.set_state_machine("IDLE")
        self._node.publish_arm_status()
        goal_handle.succeed()
        return result

    def execute_follow_joint_trajectory(self, goal_handle):
        goal = goal_handle.request
        result = FollowJointTrajectory.Result()
        trajectory = goal.trajectory
        joint_names = list(trajectory.joint_names)

        if not joint_names or not trajectory.points:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "trajectory must include joint_names and points"
            return result

        if joint_names != self._hardware.joint_names:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = (
                f"trajectory joint_names must be {self._hardware.joint_names}"
            )
            return result

        targets = [np.array(point.positions, dtype=np.float64) for point in trajectory.points]
        if any(len(target) != len(self._hardware.joint_names) for target in targets):
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "point positions length must match joint_names"
            return result

        try:
            self._hardware.start_endpos_control()
            self._hardware.set_state_machine("TRAJ_RUNNING")
            self._node.publish_arm_status()
            start = time.monotonic()
            point_times = [
                float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) * 1e-9
                for point in trajectory.points
            ]
            if point_times[0] > 0.0:
                targets.insert(0, self._hardware.get_joint_positions().copy())
                point_times.insert(0, 0.0)

            for index in range(1, len(targets)):
                q0 = targets[index - 1]
                q1 = targets[index]
                t0 = point_times[index - 1]
                t1 = max(point_times[index], t0)
                desired_velocities = np.zeros_like(q1)

                while True:
                    now = time.monotonic() - start
                    ratio = 1.0 if t1 <= t0 else max(0.0, min(1.0, (now - t0) / (t1 - t0)))
                    target = q0 + (q1 - q0) * ratio
                    self._hardware.set_joint_position_target(target)

                    positions = self._hardware.get_joint_positions()
                    velocities = self._hardware.get_joint_velocities()
                    feedback = FollowJointTrajectory.Feedback()
                    feedback.header.stamp = self._node.get_clock().now().to_msg()
                    feedback.joint_names = self._hardware.joint_names
                    feedback.desired.positions = [float(v) for v in target]
                    feedback.desired.velocities = [float(v) for v in desired_velocities]
                    feedback.actual.positions = [float(v) for v in positions]
                    feedback.actual.velocities = [float(v) for v in velocities]
                    feedback.error.positions = [float(v) for v in target - positions]
                    feedback.error.velocities = [
                        float(v) for v in desired_velocities - velocities
                    ]
                    goal_handle.publish_feedback(feedback)

                    if goal_handle.is_cancel_requested:
                        self._hardware.hold_current_position()
                        goal_handle.canceled()
                        result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                        result.error_string = "follow_joint_trajectory canceled"
                        return result

                    if now >= t1:
                        break
                    time.sleep(0.02)

        except Exception as exc:
            self._hardware.hold_current_position()
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(exc)
            return result
        finally:
            self._hardware.set_state_machine("IDLE")
            self._node.publish_arm_status()

        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        positions = self._hardware.get_joint_positions()
        velocities = self._hardware.get_joint_velocities()
        result.error_string = (
            "joint target accepted "
            f"positions={[float(v) for v in positions]} "
            f"velocities={[float(v) for v in velocities]}"
        )
        return result

    def execute_gripper_command(self, goal_handle):
        goal = goal_handle.request.command
        result = GripperCommand.Result()
        feedback = GripperCommand.Feedback()

        try:
            self._hardware.set_gripper_target(goal.position)
        except Exception:
            goal_handle.abort()
            result.position = 0.0
            result.effort = 0.0
            result.stalled = False
            result.reached_goal = False
            return result

        start = time.monotonic()
        last_pos = self._hardware.get_gripper_state()[0]
        stalled = False
        while time.monotonic() - start < 5.0:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.position = self._hardware.get_gripper_state()[0]
                result.effort = self._hardware.get_gripper_state()[2]
                result.stalled = stalled
                result.reached_goal = False
                return result

            pos, _, effort, _ = self._hardware.get_gripper_state()
            reached = self._hardware.gripper_reached_target()
            stalled = abs(pos - last_pos) < 1e-4 and abs(effort) >= float(goal.max_effort)
            feedback.position = pos
            feedback.effort = effort
            feedback.stalled = stalled
            feedback.reached_goal = reached
            goal_handle.publish_feedback(feedback)
            if reached:
                break
            last_pos = pos
            time.sleep(0.05)

        result.position = self._hardware.get_gripper_state()[0]
        result.effort = self._hardware.get_gripper_state()[2]
        result.stalled = stalled
        result.reached_goal = self._hardware.gripper_reached_target()
        goal_handle.succeed()
        return result
