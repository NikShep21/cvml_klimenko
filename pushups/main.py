import cv2
import time
import threading
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo26n-pose.pt"
SOUND_PATH = BASE_DIR / "acolyteyes2.mp3"


def play_sound():
    threading.Thread(
        target=playsound,
        args=(str(SOUND_PATH),),
        daemon=True
    ).start()


def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs((ab - cb) * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def detect_pushup(annotated, keypoints, count, stage, down_y):
    right_shoulder = keypoints[6]
    right_elbow = keypoints[8]
    right_wrist = keypoints[10]

    points_seen = (
        right_shoulder[0] > 0 and right_shoulder[1] > 0 and
        right_elbow[0] > 0 and right_elbow[1] > 0 and
        right_wrist[0] > 0 and right_wrist[1] > 0
    )

    if not points_seen:
        return count, stage, down_y

    angle = get_angle(right_shoulder, right_elbow, right_wrist)

    shoulder_y = right_shoulder[1]

    if angle < 100:
        stage = "down"
        down_y = shoulder_y

    if angle > 155 and stage == "down" and down_y is not None:
        shoulders_moved_up = shoulder_y < down_y - 20

        if shoulders_moved_up:
            stage = "up"
            count += 1
            play_sound()

    cv2.putText(
        annotated,
        f"Angle: {int(angle)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Stage: {stage}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Shoulder Y: {int(shoulder_y)}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    return count, stage, down_y


model = YOLO(str(MODEL_PATH))
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
camera.set(cv2.CAP_PROP_EXPOSURE, 5)

cv2.namedWindow("Push-ups", cv2.WINDOW_NORMAL)

count = 0
stage = None
last_seen = time.time()
down_y = None

RESET_TIME = 3

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    t = time.perf_counter()
    results = model(frame)
    print(f"FPS {1 / (time.perf_counter() - t):.1f}")

    if not results:
        continue

    result = results[0]
    keypoints = result.keypoints.xy.tolist()

    annotator = Annotator(frame)

    if not keypoints:
        if time.time() - last_seen > RESET_TIME:
            count = 0
            stage = None
            down_y = None

        annotated = annotator.result()

        cv2.putText(
            annotated,
            "No person",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Push-ups: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Push-ups", annotated)
        continue

    last_seen = time.time()

    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()

    count, stage, down_y = detect_pushup(
        annotated,
        keypoints[0],
        count,
        stage,
        down_y
    )

    cv2.putText(
        annotated,
        f"Push-ups: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Push-ups", annotated)

camera.release()
cv2.destroyAllWindows()
