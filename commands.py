from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np


# Enums
class RStatusesE(Enum):
    STOP = "STOP"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    CONVERGE = "CONVERGE"


class CGesturesE(Enum):
    F = "F"
    L = "L"
    R = "R"
    VICTORY = "Victory"

# Forward declarations (fix circular imports)
class Constraint: pass


class OptimizationParameter: pass


class Move: pass


class RobotCmd(ABC):
    """Abstract base for all robot commands."""

    def __init__(self, mnmnx, status):
        self._mnmnx = mnmnx
        self._status = status

    @property
    def mnmnx(self):
        return self._mnmnx

    @mnmnx.setter
    def mnmnx(self, new_mnmnx):
        self._mnmnx = new_mnmnx

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_status):
        self._status = new_status

    @abstractmethod
    def execute(self, robot):
        raise NotImplementedError("'execute' is not implemented !")


class Move(RobotCmd):
    """Base movement command with differential drive kinematics."""

    def __init__(self, mnmnx, status, vr=0.0, vl=0.0):
        super().__init__(mnmnx, status)
        self._vr = vr
        self._vl = vl

    @property
    def vr(self):
        return self._vr

    @vr.setter
    def vr(self, new_vr):
        self._vr = new_vr  # ✅ FIXED: Direct assignment

    @property
    def vl(self):
        return self._vl

    @vl.setter
    def vl(self, new_vl):
        self._vl = new_vl  # ✅ FIXED: Direct assignment

    def compute_velocities(self, b, r):
        v = r / 2 * (self.vr + self.vl)
        omega = r / b * (self.vr - self.vl)
        return v, omega

    def update_pose(self, pose, b, r, dt=1 / 30):
        v, omega = self.compute_velocities(b, r)
        x, y, theta = pose
        theta_new = theta + omega * dt
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        return np.array([x_new, y_new, theta_new % (2 * np.pi)])

    def execute(self, robot):
        robot.status = self.status
        print(f"robot status changed to: {robot.status.value}")
        robot.pose = self.update_pose(robot.pose, robot.b, robot.r)
        print(f"robot pose changed to: {robot.pose}")
        robot.add_to_trajectory(robot.pose)


# Gesture Commands
class TurnLeft(Move):
    def __init__(self, vr=0.2, vl=0.1):
        super().__init__(CGesturesE.L, RStatusesE.TURN_LEFT, vr, vl)


class TurnRight(Move):
    def __init__(self, vr=0.1, vl=0.2):
        super().__init__(CGesturesE.R, RStatusesE.TURN_RIGHT, vr, vl)


class MoveForward(Move):
    def __init__(self, vr=0.2, vl=0.2):
        super().__init__(CGesturesE.F, RStatusesE.MOVE_FORWARD, vr, vl)


class MoveBackward(Move):
    def __init__(self, vr=-0.2, vl=-0.2):
        super().__init__(CGesturesE.F, RStatusesE.MOVE_BACKWARD, vr, vl)


class Stop(Move):
    def __init__(self, vr=0.0, vl=0.0):
        super().__init__(CGesturesE.F, RStatusesE.STOP, vr, vl)


# Optimization Commands (Dataclass version - EXCELLENT!)
@dataclass
class ConvergeGreedyTarget(RobotCmd):
    target: np.ndarray
    max_vr: float = 0.3
    max_vl: float = 0.3
    eps: float = 10e-5
    max_it: int = 1000
    dt: float = 0.05
    heading_weight: float = 0.5
    constraints: Optional[List[Constraint]] = None
    ngrid: int = 9
    mnmnx = CGesturesE.VICTORY  # ✅ Fixed for dataclass
    status = RStatusesE.CONVERGE

    def execute(self, robot):
        from optimization import constrained_greedy, OptimizationParameter
        param = OptimizationParameter(
            robot.pose, robot.b, robot.r, self.target, self.max_vr, self.max_vl,
            self.eps, self.max_it, self.dt, self.heading_weight, self.constraints
        )
        cmds, path, rejected = constrained_greedy(param)
        print(f"Greedy: {len(cmds)} commands, {rejected} rejected")
        for cmd in cmds:
            cmd.execute(robot)


@dataclass
class ConvergeTargetSoftmax(ConvergeGreedyTarget):
    ngrid: int = 15
    temperature: float = 1.0

    def execute(self, robot):
        from optimization import constrained_softmax, OptimizationParameter
        param = OptimizationParameter(
            robot.pose, robot.b, robot.r, self.target, self.max_vr, self.max_vl,
            self.eps, self.max_it, self.dt, self.heading_weight, self.constraints,
            self.ngrid, self.temperature
        )
        cmds, path, rejected = constrained_softmax(param)
        print(f"Softmax: {len(cmds)} commands, {rejected} rejected")
        for cmd in cmds:
            cmd.execute(robot)


@dataclass
class ConvergeTargetChainedGreedySoftmax(ConvergeTargetSoftmax):
    """Hybrid: Greedy → Softmax fallback when stuck."""

    def execute(self, robot):
        from optimization import chained_greedy_softmax, OptimizationParameter
        param = OptimizationParameter(
            robot.pose, robot.b, robot.r, self.target, self.max_vr, self.max_vl,
            self.eps, self.max_it, self.dt, self.heading_weight, self.constraints,
            self.ngrid, self.temperature
        )
        cmds, path, rejected = chained_greedy_softmax(param)
        print(f"Chained: {len(cmds)} commands, {rejected} rejected")
        for cmd in cmds:
            cmd.execute(robot)


class ConvergeTargetAligned(ConvergeGreedyTarget):
    """Pure pursuit controller - direct target alignment."""

    def execute(self, robot):
        current_pose = robot.pose.copy()
        for it in range(self.max_it):
            if np.linalg.norm(current_pose[:2] - self.target[:2]) < self.eps:
                break

            # Pure pursuit
            dx = self.target[0] - current_pose[0]
            dy = self.target[1] - current_pose[1]
            desired_theta = np.arctan2(dy, dx)
            theta_error = (desired_theta - current_pose[2] + np.pi) % (2 * np.pi) - np.pi

            vr = 0.25 * (1 + 2 * theta_error)
            vl = 0.25 * (1 - 2 * theta_error)
            vr = np.clip(vr, -self.max_vr, self.max_vr)
            vl = np.clip(vl, -self.max_vl, self.max_vl)

            cmd = Move(CGesturesE.VICTORY, RStatusesE.CONVERGE, vr, vl)
            cand_pose = cmd.update_pose(current_pose, robot.b, robot.r, self.dt)

            # Safety check
            if self.constraints and not all(c.check(cand_pose) for c in self.constraints):
                Stop().execute(robot)
                break

            cmd.execute(robot)
            current_pose = robot.pose

        robot.status = RStatusesE.CONVERGE


class SafeConverge(ConvergeTargetChainedGreedySoftmax):
    """Fail-safe convergence with emergency stop."""

    def execute(self, robot):
        try:
            super().execute(robot)
        except Exception as e:
            Stop().execute(robot)
            print(f"🚨 Safety stop: {e}")
