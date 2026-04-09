from typing import List, Any, Tuple

import numpy as np

from commands import Move
from constraints import Constraint


# Utilities
def lookahead_safe(cand_pose: np.ndarray, cmd: 'Move', b: float, r: float, dt: float,
                   constraints: List['Constraint'], steps: int = 3) -> bool:
    safe_pose = cand_pose.copy()
    for _ in range(steps):
        future_pose = cmd.update_pose(safe_pose, b, r, dt)
        if constraints and not all(c.check(future_pose) for c in constraints):
            return False
        safe_pose = future_pose
    return True


def prob_select(rng: np.random.Generator, temperature: float, valid_cmds: List[Any],
                valid_poses: List[np.ndarray], valid_scores: List[float]) -> Tuple[Any, np.ndarray]:
    pr_scores = np.array([1 / (s + 1e-6) for s in valid_scores])
    pr_scores = pr_scores ** (1 / temperature)
    pr_scores /= pr_scores.sum()

    choice_idx = rng.choice(len(valid_cmds), p=pr_scores)
    return valid_cmds[choice_idx], valid_poses[choice_idx]


def init_cmds(max_vl: float, max_vr: float, ngrid: int) -> List['Move']:
    from commands import Move, CGesturesE, RStatusesE
    x = np.linspace(-max_vr, max_vr, ngrid)
    y = np.linspace(-max_vl, max_vl, ngrid)
    return [Move(CGesturesE.VICTORY, RStatusesE.CONVERGE, vr, vl) for vr in x for vl in y]


def eval_dist(cand_pose: np.ndarray, heading_weight: float, target: np.ndarray) -> float:
    pos_dist = np.linalg.norm(cand_pose[:2] - target[:2])
    head_dist = min(abs(cand_pose[2] - target[2]) % np.pi, np.pi - abs(cand_pose[2] - target[2]) % np.pi)
    return pos_dist + heading_weight * head_dist
