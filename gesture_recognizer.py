import time
from abc import abstractmethod, ABC
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
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

    _deltas: list[int]

    def __init__(self, mnmnx, status, axis='z', rotation=0, translation=None):
        super().__init__(mnmnx, status)
        self._axis = axis
        self._rotation = rotation
        if translation is None:
            translation = np.array([0.0, 0.0, 0.0])
        self._translation = translation

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, new_axis):
        self._axis = new_axis

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation):
        self._rotation = new_rotation

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, new_translation):
        self._translation = new_translation

    @property
    def tf(self):
        rotation = R.from_euler(self.axis, self.rotation, degrees=True)
        return Tf.from_components(self.translation, rotation)

    def execute(self, robot):
        robot.status = self.status
        self.rotation += robot.rotation
        print(f"robot status changed to: {robot.status.value}")
        robot.rotation = self.rotation
        print(f"robot rotation changed to: {robot.rotation}")
        robot.position = self.tf.apply(robot.position)
        print(f"robot position changed to: {robot.position}")


class TurnLeft(Move):

    def __init__(self, rotation=1):
        super().__init__(CGesturesE.L, RStatusesE.TURN_LEFT, axis = 'z', rotation=rotation)

    def execute(self, robot):
        super().execute(robot)


class TurnRight(Move):

    def __init__(self, rotation=1):
        super().__init__(CGesturesE.L, RStatusesE.TURN_RIGHT, axis = 'z', rotation=-rotation)

    def execute(self, robot):
        super().execute(robot)


class MoveForward(Move):

    def __init__(self, distance):
        super().__init__(CGesturesE.F, RStatusesE.MOVE_FORWARD, translation=[distance, 0, 0])


class MoveBackward(Move):

    def __init__(self, distance):
        super().__init__(CGesturesE.F, RStatusesE.MOVE_BACKWARD, translation=[-distance, 0, 0])



class Robot:

    def __init__(self, status: RStatusesE=RStatusesE.STOP, position: Any=None, rotation: float = 0.0):
        self._status = status
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self._position = position
        self._rotation = rotation

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_status):
        self._status = new_status

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._position = new_position

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation):
        self._rotation = new_rotation


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

def test_move():
    robot1 = Robot(RStatusesE.STOP, np.array([10, 15, 0]))
    mv1 = Move('F', RStatusesE.MOVE_FORWARD, 'z', 45, np.array([1, 1, 0]))
    mv1.execute(robot1)
    print(f"Robot status after execution: {robot1.status.value}, robot position: {robot1.position}, robot rotation around z: {robot1.rotation}")

def test_turn_left():
    robot1 = Robot(RStatusesE.STOP, np.array([10, 15, 0]))
    tl =  TurnLeft(10)
    tl.execute(robot1)
    print(
        f"Robot status after execution: {robot1.status.value}, robot position: {robot1.position}, robot rotation around z: {robot1.rotation}")

def test_turn_right():
    robot1 = Robot(RStatusesE.STOP, np.array([10, 15, 0]))
    tr =  TurnRight(10)
    tr.execute(robot1)
    print(
        f"Robot status after execution: {robot1.status.value}, robot position: {robot1.position}, robot rotation around z: {robot1.rotation}")

def test_move_forward():
    robot1 = Robot(RStatusesE.STOP, np.array([10, 15, 0]))
    mf =  MoveForward(10)
    mf.execute(robot1)
    print(
        f"Robot status after execution: {robot1.status.value}, robot position: {robot1.position}, robot rotation around z: {robot1.rotation}")

def test_move_backward():
    robot1 = Robot(RStatusesE.STOP, np.array([10, 15, 0]))
    mb =  MoveBackward(10)
    mb.execute(robot1)
    print(
        f"Robot status after execution: {robot1.status.value}, robot position: {robot1.position}, robot rotation around z: {robot1.rotation}")


def test_tl_mf_tr_mb():
    robot1 = Robot(RStatusesE.STOP, np.array([10, 15, 0]))
    tl = TurnLeft(10)
    tl.execute(robot1)
    mf = MoveForward(10)
    mf.execute(robot1)
    tr = TurnRight(10)
    tr.execute(robot1)
    mb = MoveBackward(10)
    mb.execute(robot1)
    print(
        f"Robot status after execution: {robot1.status.value}, robot position: {robot1.position}, robot rotation around z: {robot1.rotation}")
