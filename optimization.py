import pdb

import numpy as np

from typing import TYPE_CHECKING, List

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from commands import Move


class Constraint(ABC):
    @abstractmethod
    def check(self, pose: np.ndarray) -> bool:
        """True if pose satisfies constraint."""
        pass


class BoxObstacleConstraint(Constraint):
    def __init__(self, x1: float, x2: float, y1: float, y2: float, margin: float = 0.01):
        self.x1, self.x2 = min(x1, x2) - margin, max(x1, x2) + margin
        self.y1, self.y2 = min(y1, y2) - margin, max(y1, y2) + margin

    def check(self, pose: np.ndarray) -> bool:
        x, y = pose[0], pose[1]
        inside = self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
        return not inside


def lookahead_safe(cand_pose: np.ndarray, cmd: "Move", b: float, r: float,
                   dt: float, constraints: List[Constraint],
                   lookahead_steps: int = 3) -> bool:
    """
    Check if command stays safe for next N steps.

    Args:
        cand_pose: Next pose after 1 step
        cmd: Current velocity command
        b, r, dt: Robot kinematics params
        constraints: List of Constraint objects
        lookahead_steps: Check N steps ahead (default=3)

    Returns:
        True if all future poses satisfy ALL constraints
    """
    # Start from candidate pose (after 1st step)
    safe_pose = cand_pose.copy()

    for step in range(lookahead_steps):
        # Simulate future pose with same command
        future_pose = cmd.update_pose(safe_pose, b, r, dt)

        # Check ALL constraints
        if constraints and not all(c.check(future_pose) for c in constraints):
            return False

        safe_pose = future_pose

    return True


def comb_const_vel_constrained_greedy(init_pose: np.ndarray, b: float, r: float, target: np.ndarray,
                               max_vr: float = 0.3, max_vl: float = 0.3, eps: float = 0.05,
                               max_it: int = 300, dt: float = 0.1, heading_weight: float = 0.5,
                               constraints: List[Constraint] = None, ngrid: int = 9):
    """
    DETERMINISTIC GREEDY version - picks BEST single command each step.
    """
    from commands import Move, CGesturesE, RStatusesE

    x = np.linspace(-max_vr, max_vr, ngrid)
    y = np.linspace(-max_vl, max_vl, ngrid)
    cmds = [Move(CGesturesE.VICTORY, RStatusesE.CONVERGE, vr, vl) for vr in x for vl in y]

    current_pose = init_pose.copy()
    opt_cmds = []
    poses_history = [current_pose.copy()]
    rejected = 0

    for it in range(max_it):
        if np.linalg.norm(current_pose[:2] - target[:2]) < eps:
            break

        best_score = np.inf
        best_pose = None
        best_cmd = None

        for cmd in cmds:
            cand_pose = cmd.update_pose(current_pose, b, r, dt)

            # Skip if ANY constraint violated (ALL must pass)
            if constraints and not all(c.check(cand_pose) for c in constraints):
                rejected += 1
                continue

            pos_dist = np.linalg.norm(cand_pose[:2] - target[:2])
            head_dist = min(abs(cand_pose[2] - target[2]) % np.pi,
                            np.pi - abs(cand_pose[2] - target[2]) % np.pi)
            score = pos_dist + heading_weight * head_dist

            if score < best_score:
                best_score = score
                best_pose = cand_pose
                best_cmd = cmd

        if best_pose is None:
            print(f'Warning: No feasible cmd at step {it}')
            break

        current_pose = best_pose
        opt_cmds.append(best_cmd)
        poses_history.append(current_pose)

    return opt_cmds, np.array(poses_history), rejected

def comb_const_vel_constrained_softmax(init_pose: np.ndarray, b: float, r: float, target: np.ndarray,
                               max_vr: float = 0.3, max_vl: float = 0.3, eps: float = 0.05,
                               max_it: int = 300, dt: float = 0.1, heading_weight: float = 0.5,
                               constraints: List[Constraint] = None, ngrid: int = 15,
                               temperature: float = 1.0):  # Add temp control
    from commands import Move, CGesturesE, RStatusesE
    x = np.linspace(-max_vr, max_vr, ngrid)
    y = np.linspace(-max_vl, max_vl, ngrid)
    cmds = [Move(CGesturesE.VICTORY, RStatusesE.CONVERGE, vr, vl) for vr in x for vl in y]

    current_pose = init_pose.copy()
    opt_cmds = []
    poses_history = [current_pose.copy()]
    rejected = 0

    rng = np.random.default_rng()

    for it in range(max_it):
        if np.linalg.norm(current_pose[:2] - target[:2]) < eps:
            break

        valid_scores = []  # Only valid commands
        valid_poses = []
        valid_cmds = []  # Track corresponding cmds

        for cmd in cmds:
            cand_pose = cmd.update_pose(current_pose, b, r, dt)

            # Skip invalid (FIX 1: handle lookahead_safe)
            if constraints and not lookahead_safe(cand_pose, cmd, b, r, dt, constraints):
                rejected += 1
                continue

            pos_dist = np.linalg.norm(cand_pose[:2] - target[:2])
            head_dist = min(abs(cand_pose[2] - target[2]) % np.pi, np.pi - abs(cand_pose[2] - target[2]) % np.pi)
            score = pos_dist + heading_weight * head_dist

            valid_scores.append(score)
            valid_poses.append(cand_pose)
            valid_cmds.append(cmd)

        if not valid_scores:  # No feasible moves
            print(f'No feasible moves at step {it}')
            break

        # PROBABILISTIC SELECTION (FIX 2: only valid cmds)
        pr_scores = np.array([1 / (s + 1e-6) for s in valid_scores])  # Avoid div0
        pr_scores = pr_scores ** (1 / temperature)  # Temperature control
        pr_scores /= pr_scores.sum()  # Normalize

        # Sample! (FIX 3: from valid indices)
        choice_idx = rng.choice(len(valid_cmds), p=pr_scores)
        best_cmd = valid_cmds[choice_idx]
        best_pose = valid_poses[choice_idx]

        current_pose = best_pose
        opt_cmds.append(best_cmd)
        poses_history.append(current_pose)

    return opt_cmds, np.array(poses_history), rejected
