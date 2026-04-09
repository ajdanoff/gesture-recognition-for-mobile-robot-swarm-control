from abc import ABC, abstractmethod

import numpy as np


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
