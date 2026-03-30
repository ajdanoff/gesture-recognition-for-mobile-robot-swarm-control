import time
from abc import abstractmethod, ABC
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pytest
from matplotlib import pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scipy.spatial.transform import RigidTransform as Tf
from scipy.spatial.transform import Rotation as R

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

from enum import Enum


class RStatusesE(Enum):
    ASCEND = "ASCEND"
    CONTROL = "CONTROL"
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
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        return np.array([x, y, theta])

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



class Robot:

    def __init__(self, robot_id, status: RStatusesE=RStatusesE.STOP, pose = None, b = 0.3, r = 0.05):
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


class SimpleGestureClassifier:

    def extract_landmarks(self, hand_landmarks):
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
        return landmarks

    def predict(self, landmarks):
        # Extract key landmarks (thumb, index, middle, ring, pinky tips)
        tips = [4, 8, 12, 16, 20]  # Landmark indices
        pips = [3, 6, 10, 14, 18]
        tip_status = []
        pip_status = []

        for tip, pip in zip(tips, pips):
            # Check if fingertip is above PIP joint (extended finger)
            # pip = tip - 2
            if landmarks[tip][1] < landmarks[pip][1]:
                tip_status.append(1)
            else:
                if landmarks[tip][1] > landmarks[pip][1]:
                    pip_status.append(1)
                else:
                    pip_status.append(0)
                tip_status.append(0)

        crossing_fingers = False
        if landmarks[12][0] > landmarks[8][0]:
            crossing_fingers = True

        extended_count = sum(tip_status)
        # Gesture classification rules
        print(tip_status)
        print(pip_status)
        if tip_status[0] == tip_status[1] == 1 and sum(tip_status[2:]) == 0:
            return 'L'
        elif tip_status[0] == tip_status[4] == 1 and sum(tip_status[1:4]) == 0:
            return 'Y'
        elif tip_status[0] == 1 and sum(tip_status[1:]) == 4:
            return 'B'
        elif tip_status[0] == 1 and tip_status[1] == 0 and sum(tip_status[2:]) == 3:
            return 'F'
        elif tip_status[0] == tip_status[1] == tip_status[2] == 1 and sum(tip_status[3:]) == 0:
            if crossing_fingers:
                return 'R'
            return 'U'
        elif tip_status[0] == tip_status[1] == tip_status[2] == tip_status[3] == 1 and tip_status[4] == 0:
            return 'W'
        elif tip_status[0] == 1 and sum(pip_status) == 4:
            return 'C|E'

        return 'unknown'


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    vision_running_mode = mp.tasks.vision.RunningMode
    options = vision.GestureRecognizerOptions(base_options=base_options,
                                              running_mode=vision_running_mode.IMAGE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            recognition_result = recognizer.recognize(mp_frame)

            # STEP 5: Process the result. In this case, visualize it.
            try:
                top_gesture = recognition_result.gestures[0][0]
                hand_landmarks = recognition_result.hand_landmarks
                if top_gesture.category_name == 'None':
                    print(f'category is {top_gesture.category_name}, predicting using landmarks: {hand_landmarks}')
                    simple_predictor = SimpleGestureClassifier()
                    landmarks = simple_predictor.extract_landmarks(hand_landmarks[0])
                    top_gesture.category_name = simple_predictor.predict(landmarks)
                metrics = draw_landmarks(frame, hand_landmarks, top_gesture)
                # frame, metrics = display_gesture_landmarks(mp_frame, top_gesture, hand_landmarks)
            except Exception as e:
                print(f"Error: {e}")
            else:
                print(f"Gesture: {top_gesture}, Metrics: {metrics}")

        cv2.imshow('Gesture Robot Swarm Control', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'gesture_demo{int(time.time())}.png', frame)
            print('Production screenshot saved !')

    cap.release()
    cv2.destroyAllWindows()


def draw_landmarks(frame, hand_landmarks, top_gesture) -> str:
    metrics = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
    for hand_landmark in hand_landmarks:
        mp_drawing.draw_landmarks(frame,
                                  hand_landmark,
                                  mp_hands.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style()
                                  )
    cv2.putText(frame, metrics, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return metrics


if __name__ == "__main__":
    main()

@pytest.mark.parametrize("cmd_cls", [
    TurnLeft,
    TurnRight,
    MoveForward,
    MoveBackward
])
def test_cmd(cmd_cls: Any):
    robot1 = Robot('robot1', RStatusesE.STOP, np.array([10, 15, 0]))
    for _ in range(100):
        cmd = cmd_cls()
        cmd.execute(robot1)
    robot1.show_trajectory()
    print(f"Robot status after execution: {robot1.status.value}, robot pose: {robot1.pose}")

def test_drive():
    robot1 = Robot('robot1', RStatusesE.STOP, np.array([0, 0, 0]))
    for _ in range(100):
        tl = TurnLeft()
        tl.execute(robot1)
        mf = MoveForward()
        mf.execute(robot1)
    st = Stop()
    st.execute(robot1)
    for _ in range(100):
        tr = TurnRight()
        tr.execute(robot1)
        mf = MoveForward()
        mf.execute(robot1)
    st = Stop()
    st.execute(robot1)
    for _ in range(100):
        tl = TurnLeft()
        tl.execute(robot1)
        mb = MoveBackward()
        mb.execute(robot1)
    print(f"Robot status after execution: {robot1.status.value}, robot pose: {robot1.pose}")
    robot1.show_trajectory()
