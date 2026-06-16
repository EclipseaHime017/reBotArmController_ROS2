#!/usr/bin/env python3
"""Interactive gripper open/close demo for the reBotArm ROS controller."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rebotarm_msgs.srv import GripperCommand
from std_srvs.srv import Trigger

_NAMESPACE = "rebotarm"


class DemoGripperControl(Node):
    def __init__(self) -> None:
        super().__init__("gripper_control")
        self._enabled_by_demo = False
        self._enable = self.create_client(Trigger, f"/{_NAMESPACE}/enable")
        self._disable = self.create_client(Trigger, f"/{_NAMESPACE}/disable")
        self._open = self.create_client(GripperCommand, f"/{_NAMESPACE}/gripper/open")
        self._close = self.create_client(GripperCommand, f"/{_NAMESPACE}/gripper/close")

    def run(self) -> bool:
        if not self._call_trigger(self._enable, "enable"):
            return False
        self._enabled_by_demo = True

        if not self._open.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("gripper open service not available")
            return False
        if not self._close.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("gripper close service not available")
            return False

        self.get_logger().info("commands: o/open, c/close, q/quit")
        while rclpy.ok():
            try:
                command = input("gripper> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if command in ("q", "quit", "exit"):
                break
            if command in ("o", "open"):
                self._call_gripper(self._open, "open")
                continue
            if command in ("c", "close"):
                self._call_gripper(self._close, "close")
                continue
            self.get_logger().info("commands: o/open, c/close, q/quit")
        return True

    def cleanup(self) -> None:
        if self._enabled_by_demo:
            self._call_gripper(self._close, "close")
            self._call_trigger(self._disable, "disable")

    def _call_trigger(self, client, label: str, timeout_sec: float = 5.0) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"{label} service not available")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            self.get_logger().error(f"{label} timed out")
            return False
        result = future.result()
        if result is None or not result.success:
            message = result.message if result is not None else "no response"
            self.get_logger().error(f"{label} failed: {message}")
            return False
        self.get_logger().info(message if (message := result.message) else f"{label} OK")
        return True

    def _call_gripper(self, client, label: str) -> bool:
        request = GripperCommand.Request()

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            self.get_logger().error(f"{label} failed: no response")
            return False
        if not result.success:
            self.get_logger().warn(
                f"{label} not reached, current={result.reached_position:.3f}rad"
            )
            return False
        self.get_logger().info(
            f"{label} reached, current={result.reached_position:.3f}rad"
        )
        return True


def main() -> None:
    rclpy.init()
    node = DemoGripperControl()
    try:
        ok = node.run()
    except Exception as exc:
        node.get_logger().error(str(exc))
        ok = False
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
