import time
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
import pytest
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from commands import TurnLeft, TurnRight, MoveForward, MoveBackward, Stop, RStatusesE, ConvergeGreedyTarget, \
    ConvergeTargetAligned, ConvergeTargetSoftmax, ConvergeTargetChainedGreedySoftmax, SafeConverge
from constraints import BoxObstacleConstraint
from robot import Robot

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


class SimpleGestureClassifier:

    def extract_landmarks(self, hand_landmarks):
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        return landmarks

    def predict(self, landmarks):
        """Classify gestures from landmark positions."""
        tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        pips = [3, 6, 10, 14, 18]  # PIP joints

        tip_status = []
        for tip, pip in zip(tips, pips):
            # Extended finger if tip ABOVE PIP (smaller Y in image coords)
            tip_status.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

        # Gesture rules (thumb=0, index=1, middle=2, ring=3, pinky=4)
        if tip_status == [1, 1, 0, 0, 0]:
            return 'L'
        elif tip_status == [1, 0, 0, 0, 1]:
            return 'Y'
        elif tip_status == [1, 1, 1, 1, 0]:
            return 'W'
        elif sum(tip_status) == 1:
            return 'THUMBS_UP'
        elif sum(tip_status) == 0:
            return 'CLOSED_FIST'
        elif sum(tip_status) >= 3:
            return 'F'  # Open hand
        return 'F'


class GestureRobotController:
    """Main controller: Camera → Gesture → Robot Command."""

    GESTURE_MAP = {
        'Closed_Fist': Stop(),
        'Open_Palm': MoveForward(),
        'Victory': SafeConverge(np.array([2.0, 1.0, 0.0])),
        'L': TurnLeft(),
        'R': TurnRight(),
        'Thumbs_Up': MoveForward(),
        'Pointing_Up': TurnLeft(),  # Placeholder
        'CLOSED_FIST': Stop(),
        'F': MoveForward(),
        'THUMBS_UP': MoveForward(),
        'L': TurnLeft(),
        'Y': Stop(),  # Placeholder
        'W': MoveForward()
    }

    def __init__(self, model_path: str = 'gesture_recognizer.task'):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # MediaPipe setup (single instance - NO memory leak)
        self.base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        self.options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)

        self.classifier = SimpleGestureClassifier()
        self.robot = Robot('gesture_controlled', RStatusesE.STOP, np.array([0., 0., 0.]))

        # Environment
        self.obstacles = [
            BoxObstacleConstraint(0.4, 0.6, 0.2, 0.3)
        ]

    def run(self):
        """Main control loop."""
        print("🎮 Gesture Swarm Control Active (q=quit, s=screenshot)")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Gesture recognition
            result = self.recognizer.recognize(mp_frame)
            command = self.process_gesture(result, frame)

            # Execute if valid command
            if command:
                command.execute(self.robot)

            # Visual feedback
            self.draw_status(frame)
            cv2.imshow('Gesture Robot Swarm Control', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'gesture_demo_{int(time.time())}.png', frame)
                print('Screenshot saved!')

        self.cleanup()

    def process_gesture(self, result, frame) -> Optional['RobotCmd']:
        """MediaPipe → Robot Command."""
        try:
            if not result.gestures or not result.hand_landmarks:
                return None

            top_gesture = result.gestures[0][0]
            gesture_name = top_gesture.category_name

            # MediaPipe recognized gesture
            if gesture_name != 'None':
                cmd = self.GESTURE_MAP.get(gesture_name, Stop())
                cv2.putText(frame, f"MediaPipe: {gesture_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                return cmd

            # Fallback: rule-based classifier
            landmarks = self.classifier.extract_landmarks(result.hand_landmarks[0])
            fallback_gesture = self.classifier.predict(landmarks)
            cmd = self.GESTURE_MAP.get(fallback_gesture, Stop())

            cv2.putText(frame, f"Fallback: {fallback_gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return cmd

        except Exception as e:
            print(f"Gesture processing error: {e}")
            return Stop()

    def draw_status(self, frame):
        """Draw robot status and trajectory preview."""
        cv2.putText(frame, f"Status: {self.robot.status.value}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"Pose: [{self.robot.pose[0]:.2f}, {self.robot.pose[1]:.2f}]",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw hand landmarks
        if hasattr(self, 'last_landmarks'):
            for hand_landmark in self.last_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        Stop().execute(self.robot)
        print("🤖 Robot stopped. System shutdown.")


# === TESTS ===
def test_show_trajectory():
    robot = Robot("SwarmBot1")
    robot.converge_to(np.array([3, 4, 0]), max_it=50)
    robot.show_trajectory()

def test_start_live_plot():
    r1 = Robot("Bot1")
    r1.start_live_plot()
    r2 = Robot("Bot2", pose=np.array([1, 0, 0]))
    r2.start_live_plot()  # Shared figs? Use subplots
    r1.converge_to(np.array([3, 3, 0]))
    r2.converge_to(np.array([3, 4, 0]))
    r1.show_trajectory()  # Static summary

def test_simple_gesture_classifier():
    """Test rule-based classifier accuracy."""
    classifier = SimpleGestureClassifier()

    # Mock landmarks for L gesture (thumb+index extended)
    mock_landmarks = np.zeros((21, 3))
    # Simulate thumb+index extended
    mock_landmarks[4, 1] = 0.3  # Thumb tip up
    mock_landmarks[3, 1] = 0.35  # Thumb PIP down
    mock_landmarks[8, 1] = 0.25  # Index tip up
    mock_landmarks[6, 1] = 0.3  # Index PIP down

    gesture = classifier.predict(mock_landmarks)
    assert gesture == 'L', f"Expected 'L', got {gesture}"


@pytest.mark.parametrize("cmd_cls", [TurnLeft, TurnRight, MoveForward, MoveBackward])
def test_basic_commands(cmd_cls: Any):
    """Test individual gesture commands."""
    robot = Robot('test_cmd', RStatusesE.STOP, np.array([0., 0., 0.]))
    cmd = cmd_cls()
    cmd.execute(robot)

    assert robot.status == cmd.status
    assert len(robot.trajectory) == 2
    print(f"✅ {cmd_cls.__name__}: {robot.pose}")


def test_drive_sequence():
    """Test complex movement sequence."""
    robot = Robot('test_drive', RStatusesE.STOP, np.array([0., 0., 0.]))

    # Turn left + forward
    for _ in range(5):
        TurnLeft().execute(robot)
        MoveForward().execute(robot)

    Stop().execute(robot)
    assert robot.status == RStatusesE.STOP
    robot.show_trajectory()


def test_convergence_with_obstacles():
    """Test obstacle avoidance + convergence."""
    robot = Robot('test_obs', RStatusesE.STOP, np.array([0., 0., np.pi / 4]))
    target = np.array([3., 1.5, 0.])
    obstacle = BoxObstacleConstraint(0.4, 0.6, 0.2, 0.3)

    cmd = SafeConverge(target, constraints=[obstacle], max_it=1000)
    cmd.execute(robot)

    # Verify obstacle avoidance
    for pose in robot.trajectory:
        assert obstacle.check(pose)

    final_dist = np.linalg.norm(robot.pose[:2] - target[:2])
    print(f"✅ Final distance: {final_dist:.3f}m")
    robot.show_trajectory()


def test_gesture_integration_pipeline():
    """End-to-end gesture → robot movement."""
    controller = GestureRobotController()
    controller.robot = Robot('test_gesture', RStatusesE.STOP, np.array([0., 0., 0.]))

    # Simulate Victory gesture
    mock_cmd = ConvergeTargetChainedGreedySoftmax(np.array([2., 1., 0.]))
    mock_cmd.execute(controller.robot)

    assert len(controller.robot.trajectory) > 10
    assert controller.robot.status == RStatusesE.CONVERGE
    print("✅ Full gesture pipeline OK")


if __name__ == "__main__":
    controller = GestureRobotController()
    controller.run()