import numpy as np
from matplotlib import pyplot as plt

from commands import RStatusesE


class Robot:

    def __init__(self, robot_id, status: RStatusesE = RStatusesE.STOP, pose = None, b = 0.3, r = 0.05):
        self._robot_id = robot_id
        self._status = status
        if pose is None:
            pose = np.array([0.0, 0.0, 0.0])
        self._pose = pose
        self._b = b
        self._r = r
        self._trajectory = np.array([pose])

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, new_b):
        self._b = new_b

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, new_r):
        self._r = new_r

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_status):
        self._status = new_status

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, new_pose):
        self._pose = new_pose

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def robot_id(self):
        return self._robot_id

    @robot_id.setter
    def robot_id(self, new_robot_id):
        self._robot_id = new_robot_id

    def add_to_trajectory(self, point):
        self._trajectory = np.concatenate((self.trajectory, [point]), axis=0)

    def clear_trajectory(self):
        self._trajectory = np.array([])

    def show_trajectory(self):
        plt.figure(figsize=(8, 5))
        x = [p[0] for p in self.trajectory]
        y = [p[1] for p in self.trajectory]
        plt.plot(x, y, label=f'Trajectory at {self._robot_id}°')  # Use plt.plot for lines
        plt.title('Robot Trajectory')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.legend()
        plt.show()
