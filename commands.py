from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from optimization import comb_const_vel_constrained_greedy, Constraint, comb_const_vel_constrained_softmax


class RobotCmd(ABC):

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

    _vr: float
    _vl: float

    def __init__(self, mnmnx, status, vr=0.0, vl=0.0):
        super().__init__(mnmnx, status)
        self._vr = vr
        self._vl = vl

    @property
    def vr(self):
        return self._vr

    @vr.setter
    def vr(self, new_vr):
        self.vr = new_vr

    @property
    def vl(self):
        return self._vl

    @vl.setter
    def vl(self, new_vl):
        self.vl = new_vl

    def compute_velocities(self, b, r):
        v = r / 2 * (self.vr + self.vl)
        omega = r / b * (self.vr - self.vl)
        return  v, omega

    def update_pose(self, pose, b, r, dt=1/30):
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


class TurnLeft(Move):

    def __init__(self, vr=0.2, vl=0.1):
        super().__init__(CGesturesE.L, RStatusesE.TURN_LEFT, vr, vl)

    def execute(self, robot):
        super().execute(robot)


class TurnRight(Move):

    def __init__(self, vr=0.1, vl=0.2):
        super().__init__(CGesturesE.R, RStatusesE.TURN_RIGHT, vr, vl)

    def execute(self, robot):
        super().execute(robot)


class MoveForward(Move):

    def __init__(self, vr=0.2, vl=0.2):
        super().__init__(CGesturesE.F, RStatusesE.MOVE_FORWARD, vr, vl)


class MoveBackward(Move):

    def __init__(self, vr=-0.2, vl=-0.2):
        super().__init__(CGesturesE.F, RStatusesE.MOVE_BACKWARD, vr, vl)


class Stop(Move):

    def __init__(self, vr=0.0, vl=0.0):
        super().__init__(CGesturesE.F, RStatusesE.STOP, vr, vl)


class ConvergeGreedyTarget(RobotCmd):

    def __init__(self,
                 target: np.ndarray,
                 max_vr: float = 0.3,
                 max_vl: float = 0.3,
                 eps: float = 10e-5,
                 max_it: int = 1000,
                 dt: float = 0.05,
                 heading_weight: float = 0.5,
                 constraints: list[Constraint] = None
                 ):
        super().__init__(CGesturesE.VICTORY, RStatusesE.CONVERGE)
        self._target = target
        self._max_it = max_it
        self._max_vr = max_vr
        self._max_vl = max_vl
        self._eps = eps
        self._dt = dt
        self._heading_weight = heading_weight
        self._constraints = constraints

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, new_constraints):
        self._constraints = new_constraints

    @property
    def heading_weight(self):
        return self._heading_weight

    @heading_weight.setter
    def heading_weight(self, new_heading_weight):
        self._heading_weight = new_heading_weight

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, new_dt):
        self._dt = new_dt

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, new_target):
        self._target = new_target

    @property
    def max_it(self):
        return self._max_it

    @max_it.setter
    def max_it(self, new_maxit):
        self._max_it = new_maxit

    @property
    def max_vr(self):
        return self._max_vr

    @max_vr.setter
    def max_vr(self, new_max_vr):
        self._max_vr = new_max_vr

    @property
    def max_vl(self):
        return self._max_vl

    @max_vl.setter
    def max_vl(self, new_max_vl):
        self._max_vl = new_max_vl

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, new_eps):
        self._eps = new_eps

    def execute(self, robot):
        cmds, path, rejected = comb_const_vel_constrained_greedy(
            robot.pose,
            robot.b,
            robot.r,
            self.target,
            self.max_vr,
            self.max_vl,
            self.eps,
            self.max_it,
            self.dt,
            self.heading_weight,
            self.constraints
        )
        for cmd in cmds:
            cmd.execute(robot)


class ConvergeTargetSoftmax(ConvergeGreedyTarget):

    def __init__(self,
                 target: np.ndarray,
                 max_vr: float = 0.3,
                 max_vl: float = 0.3,
                 eps: float = 10e-5,
                 max_it: int = 1000,
                 dt: float = 0.05,
                 heading_weight: float = 0.5,
                 constraints: list[Constraint] = None,
                 temperature: float = 1.0
                 ):
        super().__init__(
            target,
            max_vr,
            max_vl,
            eps,
            max_it,
            dt,
            heading_weight,
            constraints
        )
        self._temperature = temperature

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, new_temperature):
        self._temperature = new_temperature

    def execute(self, robot):
        cmds, path, rejected = comb_const_vel_constrained_softmax(
            robot.pose,
            robot.b,
            robot.r,
            self.target,
            self.max_vr,
            self.max_vl,
            self.eps,
            self.max_it,
            self.dt,
            self.heading_weight,
            self.constraints
        )
        for cmd in cmds:
            cmd.execute(robot)


class ConvergeTargetAligned(ConvergeGreedyTarget):

    def execute(self, robot):
        # ... setup ...
        current_pose = robot.pose
        for it in range(self.max_it):
            if np.linalg.norm(current_pose[:2] - self.target[:2]) < self.eps:
                break

            # Direct line to target
            dx = self.target[0] - current_pose[0]
            dy = self.target[1] - current_pose[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            desired_theta = np.arctan2(dy, dx)

            # Align heading + forward
            theta_error = desired_theta - current_pose[2]
            theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi

            vr = 0.25 * (1 + 2 * theta_error)  # Right faster if left error
            vl = 0.25 * (1 - 2 * theta_error)  # Left faster if right error

            # Cap velocities
            vr = np.clip(vr, -self.max_vr, self.max_vr)
            vl = np.clip(vl, -self.max_vl, self.max_vl)

            cmd = Move(CGesturesE.VICTORY, RStatusesE.CONVERGE, vr, vl)
            cand_pose = cmd.update_pose(current_pose, robot.b, robot.r)

            # Constraints check...
            if self.constraints and not all(c.check(cand_pose) for c in self.constraints):
                vr = vl = 0  # Stop if unsafe
            else:
                cmd.execute(robot)
                #current_pose = cand_pose
                #robot.trajectory.append(current_pose.copy())

        robot.pose = current_pose
        robot.status = RStatusesE.CONVERGE



class RStatusesE(Enum):
    ASCEND = "ASCEND"
    CONTROL = "CONTROL"
    CONVERGE = "CONVERGE"
    DESCEND = "DESCEND"
    DOCKING = "DOCK"
    FOLLOWING = "FOLLOW"
    GRIP = "GRIP"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    RELEASE = "RELEASE"
    STOP = "STOP"


class CGesturesE(Enum):
    UNRECOGNIZED = "Unknown"
    CLOSED_FIST = "Closed_Fist"
    OPEN_PALM = "Open_Palm"
    POINTING_UP = "Pointing_Up"
    THUMB_DOWN = "Thumb_Down"
    THUMBS_UP = "Thumb_Up"
    VICTORY = "Victory"
    LOVE = "ILoveYou"
    L = "L"
    Y = "Y"
    B = "B"
    C_E = "C|E"
    F = "F"
    U = "U"
    R = "R"
    W = "W"
