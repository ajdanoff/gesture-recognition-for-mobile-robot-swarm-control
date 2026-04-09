from typing import TYPE_CHECKING

from constraints import Constraint
from utils import lookahead_safe, prob_select, init_cmds, eval_dist

if TYPE_CHECKING:
    pass

from dataclasses import dataclass
from typing import Callable, List, Any, Tuple
import numpy as np
from numpy.random import default_rng

OptOp = Callable[['OptimizationParameter'], Tuple[List[Any], np.ndarray, int]]


@dataclass
class OptimizationParameter:
    init_pose: np.ndarray
    b: float
    r: float
    target: np.ndarray
    max_vr: float = 0.3
    max_vl: float = 0.3
    eps: float = 0.05
    max_it: int = 300
    dt: float = 0.1
    heading_weight: float = 0.5
    constraints: List['Constraint'] = None
    ngrid: int = 9
    temperature: float = 1.0
    nobest_cb: OptOp | None = None
    nobest_data: 'OptimizationParameter' = None


def constrained_greedy(opt_param: OptimizationParameter) -> Tuple[List[Any], np.ndarray, int]:
    """DETERMINISTIC GREEDY - picks BEST command each step."""
    cmds = init_cmds(opt_param.max_vl, opt_param.max_vr, opt_param.ngrid)

    current_pose = opt_param.init_pose.copy()
    opt_cmds = []
    poses_history = [current_pose.copy()]
    rejected = 0

    for it in range(opt_param.max_it):
        if np.linalg.norm(current_pose[:2] - opt_param.target[:2]) < opt_param.eps:
            break

        best_score = np.inf
        best_pose = None
        best_cmd = None

        for cmd in cmds:
            cand_pose = cmd.update_pose(current_pose, opt_param.b, opt_param.r, opt_param.dt)

            if opt_param.constraints and not all(c.check(cand_pose) for c in opt_param.constraints):
                rejected += 1
                continue

            score = eval_dist(cand_pose, opt_param.heading_weight, opt_param.target)

            if score < best_score:
                best_score = score
                best_pose = cand_pose
                best_cmd = cmd

        if best_pose is None:
            print(f'No feasible cmd at step {it}')
            if opt_param.nobest_cb:
                log_fallback(opt_param)
                cb_cmds, cb_poses, cb_rejected = opt_param.nobest_cb(opt_param.nobest_data)
                opt_cmds.extend(cb_cmds)
                poses_history = np.append(poses_history, cb_poses, axis=0)
                rejected += cb_rejected
            else:
                break

        current_pose = best_pose
        opt_cmds.append(best_cmd)
        poses_history.append(current_pose)

    return opt_cmds, np.array(poses_history), rejected


def constrained_softmax(opt_param: OptimizationParameter) -> Tuple[List[Any], np.ndarray, int]:
    """PROBABILISTIC - softmax selection from valid commands."""
    cmds = init_cmds(opt_param.max_vl, opt_param.max_vr, opt_param.ngrid)

    current_pose = opt_param.init_pose.copy()
    opt_cmds = []
    poses_history = [current_pose.copy()]
    rejected = 0

    rng = default_rng()

    for it in range(opt_param.max_it):
        if np.linalg.norm(current_pose[:2] - opt_param.target[:2]) < opt_param.eps:
            break

        valid_scores, valid_poses, valid_cmds = [], [], []

        for cmd in cmds:
            cand_pose = cmd.update_pose(current_pose, opt_param.b, opt_param.r, opt_param.dt)

            if opt_param.constraints and not lookahead_safe(cand_pose, cmd, opt_param.b, opt_param.r,
                                                            opt_param.dt, opt_param.constraints):
                rejected += 1
                continue

            score = eval_dist(cand_pose, opt_param.heading_weight, opt_param.target)
            valid_scores.append(score)
            valid_poses.append(cand_pose)
            valid_cmds.append(cmd)

        if not valid_scores:
            print(f'No feasible moves at step {it}')
            break

        best_cmd, best_pose = prob_select(rng, opt_param.temperature, valid_cmds, valid_poses, valid_scores)

        current_pose = best_pose
        opt_cmds.append(best_cmd)
        poses_history.append(current_pose)

    return opt_cmds, np.array(poses_history), rejected

def log_fallback(param: OptimizationParameter):
    print(f"Greedy stuck → Softmax fallback: ngrid={param.ngrid}, temp={param.temperature}")

def chained_greedy_softmax(param: OptimizationParameter) -> Tuple[List[Any], np.ndarray, int]:
    """
    Hybrid: Greedy → Softmax fallback when stuck.
    """
    # Create softmax params (copy + higher exploration)
    fallback_param = OptimizationParameter(
        init_pose=param.init_pose,
        b=param.b,
        r=param.r,
        target=param.target,
        max_vr=param.max_vr,
        max_vl=param.max_vl,
        eps=param.eps,
        max_it=param.max_it // 2,  # Half iterations for fallback
        dt=param.dt,
        heading_weight=param.heading_weight,
        constraints=param.constraints,
        ngrid=max(12, param.ngrid + 3),  # Coarser grid for exploration
        temperature=1.5,  # Higher exploration
        nobest_cb = None,
        nobest_data = None
    )

    # Set softmax as fallback
    param.nobest_cb = constrained_softmax
    param.nobest_data = fallback_param

    return constrained_greedy(param)


def make_hybrid_optimizer(primary: OptOp, fallback: OptOp,
                          fallback_ngrid_boost: int = 3,
                          fallback_temp: float = 1.5) -> OptOp:
    """Factory for chained optimizers."""

    def hybrid(param: OptimizationParameter) -> Tuple[List[Any], np.ndarray, int]:
        # Fallback params
        fallback_param = OptimizationParameter(
            init_pose=param.init_pose, b=param.b, r=param.r, target=param.target,
            max_vr=param.max_vr, max_vl=param.max_vl, eps=param.eps,
            max_it=param.max_it // 2, dt=param.dt, heading_weight=param.heading_weight,
            constraints=param.constraints,
            ngrid=param.ngrid + fallback_ngrid_boost,
            temperature=fallback_temp,
            nobest_cb=None, nobest_data=None
        )

        param.nobest_cb = fallback
        param.nobest_data = fallback_param

        return primary(param)

    return hybrid
