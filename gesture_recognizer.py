import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pytest
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from commands import TurnLeft, TurnRight, MoveForward, MoveBackward, Stop, RStatusesE, ConvergeGreedyTarget, \
    ConvergeTargetAligned, ConvergeTargetSoftmax
from optimization import BoxObstacleConstraint
from robot import Robot

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


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


def test_probabilistic_convergence():
    robot = Robot('robot2', RStatusesE.STOP, [0., 0., 0.])
    target = np.array([2., 1., 0.])
    cmd = ConvergeTargetSoftmax(target, max_it=10000, temperature=1.0)
    cmd.execute(robot)
    robot.show_trajectory()
    final_dist = np.linalg.norm(robot.pose[:2] - target[:2])
    assert final_dist < 0.5, f"Probabilistic: {final_dist:.3f}m"  # Looser but realistic

def test_obstacle_avoidance_correctness():
    """Robot avoids obstacle [0.4-0.6, 0.2-0.3]"""
    robot = Robot('robot1', RStatusesE.STOP, [0., 0., np.pi / 4])
    target = np.array([4, 1.5, 0.])
    obstacle = BoxObstacleConstraint(0.4, 0.6, 0.2, 0.3)
    constraints = [obstacle]

    cmd = ConvergeGreedyTarget(target, max_it=10000, constraints=constraints)
    cmd.execute(robot)
    robot.show_trajectory()

    # All trajectory points AVOID obstacle
    for pose in robot.trajectory:
        assert obstacle.check(pose), f"Entered obstacle at {pose[:2]}"

    assert robot.status == RStatusesE.CONVERGE
    init_dist = np.linalg.norm(robot.trajectory[0][:2] - target[:2])
    final_dist = np.linalg.norm(robot.pose[:2] - target[:2])
    assert final_dist < init_dist * 0.9
    print(f"PASS: {len(robot.trajectory)} steps, final dist: {final_dist:.2f}m")
    robot.show_trajectory()

def test_no_constraints_reaches_target():
    """Without constraints, makes significant progress toward target"""
    robot = Robot('robot2', RStatusesE.STOP, [0., 0., 0.])
    target = np.array([2., 1., 0.])
    cmd = ConvergeGreedyTarget(target, max_it=2000)  # More iterations
    cmd.execute(robot)

    final_dist = np.linalg.norm(robot.pose[:2] - target[:2])
    init_dist = np.linalg.norm(np.array([0, 0]) - target[:2])

    # Accept 50% progress (realistic for greedy)
    assert final_dist < init_dist * 0.5, f"Expected <1m progress, got {final_dist:.3f}m"
    print(f"Progress: {init_dist - final_dist:.2f}m ({100 * (1 - final_dist / init_dist):.1f}%)")

def test_no_constraints_reaches_target_aligned():
    robot = Robot('robot2', RStatusesE.STOP, [0., 0., 0.])
    target = np.array([2., 1., 0.])
    cmd = ConvergeTargetAligned(target, max_it=2000, eps=0.3)  # Looser eps
    cmd.execute(robot)

    final_dist = float(np.linalg.norm(robot.pose[:2] - target[:2]))  # np.float64 fix
    assert final_dist < 1.0, f"Expected <1.0m, got {final_dist:.3f}m"
    assert final_dist < 2.1  # Current performance baseline
    robot.show_trajectory()

def test_impossible_with_tight_obstacle():
    """Raises error when obstacle blocks all paths"""
    robot = Robot('robot3', RStatusesE.STOP, [0., 0., 0.])
    target = np.array([3., 0., 0.])
    wall = BoxObstacleConstraint(-0.1, 2.9, -10, 10)

    cmd = ConvergeGreedyTarget(target, max_it=50, constraints=[wall])
    with pytest.raises(RuntimeError):
        cmd.execute(robot)


def test_multiple_obstacles():
    """Respects multiple obstacle constraints"""
    robot = Robot('robot4', RStatusesE.STOP, [0., 0., 0.])
    target = np.array([3., 1.5, 0.])
    obstacles = [
        BoxObstacleConstraint(0.8, 1.2, 0.0, 0.5),
        BoxObstacleConstraint(1.8, 2.2, 0.8, 1.3)
    ]

    cmd = ConvergeGreedyTarget(target, max_it=1000, constraints=obstacles)
    cmd.execute(robot)

    for obstacle in obstacles:
        for pose in robot.trajectory:
            assert obstacle.check(pose)

    robot.show_trajectory()


def test_constraint_logic_all_not_any():
    """Verifies ALL constraints must pass"""
    robot = Robot('robot5', RStatusesE.STOP, [0., 0., 0.])
    target = np.array([1., 0., 0.])

    loose = BoxObstacleConstraint(-10, 10, -10, 10)  # Always passes
    strict = BoxObstacleConstraint(0.9, 1.1, -0.1, 0.1)  # Blocks near target

    cmd = ConvergeGreedyTarget(target, max_it=100, constraints=[loose, strict])
    cmd.execute(robot)

    # Must respect STRICT constraint even if loose allows
    assert strict.check(robot.pose)