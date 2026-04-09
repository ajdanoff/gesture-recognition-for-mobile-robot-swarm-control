import numpy as np
from matplotlib import pyplot as plt

from commands import RStatusesE, RobotCmd
import plotly.graph_objects as go


class Robot:

    def __init__(self, robot_id, status: RStatusesE = RStatusesE.STOP, pose = None, b = 0.3, r = 0.05):
        self.current_trace_idx = None
        self.path_trace_idx = None
        self._robot_id = robot_id
        self._status = status
        if pose is None:
            pose = np.array([0.0, 0.0, 0.0])
        self._pose = pose
        self._b = b
        self._r = r
        self._trajectory = np.array([pose])
        self.fig = None
        self.line = None

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

    def execute_command(self, cmd: RobotCmd):
        """Execute gesture command"""
        cmd.execute(self)

    def converge_to(self, target: np.ndarray, strategy: str = "chained", **kwargs):
        """High-level convergence"""
        from commands import ConvergeTargetChainedGreedySoftmax
        cmd = ConvergeTargetChainedGreedySoftmax(target, **kwargs)
        cmd.execute(self)

    def show_trajectory(self):
        import numpy as np
        import matplotlib.pyplot as plt
        traj = np.array(self.trajectory)
        if len(traj) == 0:
            print("No trajectory data")
            return
        plt.figure(figsize=(8, 6))
        plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Path')
        plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
        plt.plot(self.pose[0], self.pose[1], 'ro', markersize=12, label='Current')
        # Orientation arrows every 20 points
        step = max(1, len(traj) // 20)
        for i in range(0, len(traj), step):
            dx = 0.15 * np.cos(traj[i, 2])
            dy = 0.15 * np.sin(traj[i, 2])
            plt.arrow(traj[i, 0], traj[i, 1], dx, dy, head_width=0.08, fc='m', ec='m', alpha=0.7)
        plt.axis('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Robot {self.robot_id} Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def start_live_plot(self):
        self.fig = go.FigureWidget()
        # Store trace indices instead of objects
        self.path_trace_idx = 0
        self.current_trace_idx = 1

        # Path trace (index 0)
        self.fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='lines', name=f'{self.robot_id} Path',
            line=dict(color='blue', width=3)
        ))

        # Current position trace (index 1)
        self.fig.add_trace(go.Scatter(
            x=[], y=[], mode='markers+text',
            marker=dict(color='red', size=12),
            name='Current', text=[], textposition='top center'
        ))

        self.fig.update_layout(
            title=f'Live {self.robot_id} Trajectory',
            xaxis_title='X (m)', yaxis_title='Y (m)'
        )
        self.fig.show()

    def update_live(self):
        if self.fig is None or not hasattr(self, 'path_trace_idx'):
            return

        traj = np.array(self.trajectory)
        with self.fig.batch_update():
            # Use indices directly (no .id)
            self.fig.data[self.path_trace_idx].x = traj[:, 0].tolist()
            self.fig.data[self.path_trace_idx].y = traj[:, 1].tolist()

            # Single current marker
            self.fig.data[self.current_trace_idx].x = [self.pose[0]]
            self.fig.data[self.current_trace_idx].y = [self.pose[1]]
            self.fig.data[self.current_trace_idx].text = [f'{self.robot_id}']

    def clear_trajectory(self):
        self._trajectory = np.array([self.pose])  # Retain current pose
        if self.fig is not None:
            self.update_live()

    def add_to_trajectory(self, point):
        self._trajectory = np.concatenate((self.trajectory, [point]), axis=0)
        if hasattr(self, 'fig') and self.fig is not None:
            self.update_live()
