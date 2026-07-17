import json
import time
import math
import os
import platform
import urllib.request
from dataclasses import dataclass, asdict

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib"),
)

import cv2
from enum import IntEnum


class _PoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


OBSERVATION_INTERVAL_SEC  = 5.0
NO_MOVEMENT_TIMEOUT_SEC   = 10.0
CAMERA_INDEX              = 0
SHOW_WINDOW               = True

HEAD_MOVEMENT_THRESHOLD_DEG = 2.5
WRIST_MOVEMENT_THRESHOLD = 0.05   # normalized landmark distance (~5% of frame)
WRIST_VISIBILITY_MIN = 0.5
FACE_MOVEMENT_THRESHOLD = 0.025   # normalized face-center shift (~2.5% of frame)
FACE_SIZE_CHANGE_THRESHOLD = 0.08 # relative face-box area change

GAZE_ON_ROBOT_MAX_YAW_DEG   = 20.0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR  = os.path.join(_SCRIPT_DIR, "models")

_FACE_MODEL = os.path.join(_MODEL_DIR, "face_landmarker.task")
_POSE_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")

_MODEL_URLS = {
    _FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    _POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

_DEFAULT_VISION_BACKEND = (
    "opencv" if platform.system() == "Darwin" else "mediapipe_tasks"
)
_VISION_BACKEND = os.environ.get(
    "PIPELINE4_VISION_BACKEND",
    _DEFAULT_VISION_BACKEND,
).strip().lower()


def _ensure_models():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for path, url in _MODEL_URLS.items():
        if not os.path.exists(path):
            name = os.path.basename(path)
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  Done ({size_mb:.1f} MB)")


class _NoopLandmarker:
    def close(self):
        pass


def _create_mediapipe_landmarkers():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    _ensure_models()

    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=_FACE_MODEL,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    pose_landmarker_instance = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=_POSE_MODEL,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return mp, face, pose_landmarker_instance


_mp = None
_FACE_CASCADE = None
_PROFILE_FACE_CASCADE = None

if _VISION_BACKEND == "mediapipe_tasks":
    _mp, face_landmarker, pose_landmarker = _create_mediapipe_landmarkers()
else:
    face_landmarker = _NoopLandmarker()
    pose_landmarker = _NoopLandmarker()
    _FACE_CASCADE = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    _PROFILE_FACE_CASCADE = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
    )


@dataclass
class Observation:
    timestamp:          str   = ""
    face_detected:      bool  = False
    gaze_on_robot:      float = 0.0
    head_yaw_deg:       float = 0.0
    head_pitch_deg:     float = 0.0
    head_moving:        bool  = False
    hand_raised:        bool  = False
    hand_raise_side:    str   = "none"
    body_detected:      bool  = False
    no_movement_sec:    float = 0.0
    still_there_prompt: bool  = False
    speech_energy:      str   = "low"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


face_mesh = face_landmarker
pose = pose_landmarker


def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def estimate_head_pose(landmarks, img_w: int, img_h: int) -> tuple[float, float]:
    NOSE_TIP        = 1
    CHIN            = 152
    LEFT_EYE_OUTER  = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH      = 61
    RIGHT_MOUTH     = 291

    def lm(idx):
        l = landmarks[idx]
        return l.x * img_w, l.y * img_h

    nose_x,  nose_y  = lm(NOSE_TIP)
    chin_x,  chin_y  = lm(CHIN)
    leye_x,  _       = lm(LEFT_EYE_OUTER)
    reye_x,  _       = lm(RIGHT_EYE_OUTER)
    lmouth_x, _      = lm(LEFT_MOUTH)
    rmouth_x, _      = lm(RIGHT_MOUTH)

    eye_mid_x = (leye_x + reye_x) / 2.0
    eye_span  = abs(reye_x - leye_x)
    yaw_raw   = (nose_x - eye_mid_x) / (eye_span + 1e-6)
    yaw_deg   = yaw_raw * 90.0

    face_height = abs(chin_y - nose_y)
    frame_mid_y = img_h / 2.0
    pitch_raw   = (frame_mid_y - nose_y) / (face_height + 1e-6)
    pitch_deg   = pitch_raw * 30.0

    return round(yaw_deg, 1), round(pitch_deg, 1)


def estimate_gaze(yaw_deg: float) -> float:
    abs_yaw = abs(yaw_deg)
    if abs_yaw >= GAZE_ON_ROBOT_MAX_YAW_DEG:
        return 0.0
    return round(1.0 - (abs_yaw / GAZE_ON_ROBOT_MAX_YAW_DEG), 2)


def detect_hand_raise(pose_landmarks) -> tuple[bool, str]:
    PoseLM = _PoseLandmark

    l_wrist    = pose_landmarks[PoseLM.LEFT_WRIST]
    r_wrist    = pose_landmarks[PoseLM.RIGHT_WRIST]
    l_shoulder = pose_landmarks[PoseLM.LEFT_SHOULDER]
    r_shoulder = pose_landmarks[PoseLM.RIGHT_SHOULDER]

    MIN_VIS = 0.5
    l_vis = getattr(l_wrist, 'visibility', 1.0) or 1.0
    r_vis = getattr(r_wrist, 'visibility', 1.0) or 1.0
    ls_vis = getattr(l_shoulder, 'visibility', 1.0) or 1.0
    rs_vis = getattr(r_shoulder, 'visibility', 1.0) or 1.0

    left_raised  = (l_vis > MIN_VIS and ls_vis > MIN_VIS and
                    l_wrist.y < l_shoulder.y)
    right_raised = (r_vis > MIN_VIS and rs_vis > MIN_VIS and
                    r_wrist.y < r_shoulder.y)

    if left_raised and right_raised:
        return True, "both"
    elif left_raised:
        return True, "left"
    elif right_raised:
        return True, "right"
    else:
        return False, "none"


class MovementTracker:

    def __init__(self):
        self._last_movement_time = time.time()
        self._prev_yaw   = 0.0
        self._prev_pitch = 0.0
        self._prev_hand_raised = False
        self._prev_left_wrist: tuple[float, float] | None = None
        self._prev_right_wrist: tuple[float, float] | None = None
        self._prev_face_center: tuple[float, float] | None = None
        self._prev_face_size: float | None = None

    def update(self, yaw: float, pitch: float,
               hand_raised: bool, face_detected: bool,
               left_wrist: tuple[float, float] | None = None,
               right_wrist: tuple[float, float] | None = None,
               face_center: tuple[float, float] | None = None,
               face_size: float | None = None,
               ) -> tuple[bool, float]:
        now = time.time()

        yaw_delta   = abs(yaw - self._prev_yaw)
        pitch_delta = abs(pitch - self._prev_pitch)
        head_moving = (yaw_delta > HEAD_MOVEMENT_THRESHOLD_DEG or
                       pitch_delta > HEAD_MOVEMENT_THRESHOLD_DEG)

        wrist_moving = False
        if left_wrist is not None and self._prev_left_wrist is not None:
            dx = left_wrist[0] - self._prev_left_wrist[0]
            dy = left_wrist[1] - self._prev_left_wrist[1]
            if (dx * dx + dy * dy) ** 0.5 > WRIST_MOVEMENT_THRESHOLD:
                wrist_moving = True
        if (not wrist_moving and right_wrist is not None
                and self._prev_right_wrist is not None):
            dx = right_wrist[0] - self._prev_right_wrist[0]
            dy = right_wrist[1] - self._prev_right_wrist[1]
            if (dx * dx + dy * dy) ** 0.5 > WRIST_MOVEMENT_THRESHOLD:
                wrist_moving = True

        face_moving = False
        if face_center is not None:
            if self._prev_face_center is None:
                face_moving = True
            else:
                dx = face_center[0] - self._prev_face_center[0]
                dy = face_center[1] - self._prev_face_center[1]
                if (dx * dx + dy * dy) ** 0.5 > FACE_MOVEMENT_THRESHOLD:
                    face_moving = True

            if face_size is not None and self._prev_face_size:
                rel_change = abs(face_size - self._prev_face_size) / self._prev_face_size
                if rel_change > FACE_SIZE_CHANGE_THRESHOLD:
                    face_moving = True

        movement_detected = head_moving or wrist_moving or face_moving

        if movement_detected:
            self._last_movement_time = now

        self._prev_yaw         = yaw
        self._prev_pitch       = pitch
        self._prev_hand_raised = hand_raised
        self._prev_left_wrist  = left_wrist
        self._prev_right_wrist = right_wrist
        self._prev_face_center = face_center
        self._prev_face_size   = face_size

        no_movement_sec = round(now - self._last_movement_time, 1)
        return movement_detected, no_movement_sec

    def reset(self):
        self._last_movement_time = time.time()
        self._prev_face_center = None
        self._prev_face_size = None


class ObservationBuilder:

    def __init__(self):
        self._last_obs_time = time.time()
        self._frames: list[dict] = []

    def add_frame(self, frame_data: dict):
        self._frames.append(frame_data)

    def ready(self) -> bool:
        return (time.time() - self._last_obs_time) >= OBSERVATION_INTERVAL_SEC

    def build(self) -> Observation:
        self._last_obs_time = time.time()
        frames = self._frames
        self._frames = []

        obs = Observation()
        obs.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        if not frames:
            return obs

        face_frames = [f for f in frames if f.get("face_detected")]
        obs.face_detected = len(face_frames) > len(frames) * 0.5

        if face_frames:
            obs.gaze_on_robot = round(
                sum(f["gaze"] for f in face_frames) / len(face_frames), 2)

            obs.head_yaw_deg   = face_frames[-1]["yaw"]
            obs.head_pitch_deg = face_frames[-1]["pitch"]

            obs.head_moving = any(f.get("head_moving") for f in face_frames)

        body_frames = [f for f in frames if f.get("body_detected")]
        obs.body_detected = len(body_frames) > len(frames) * 0.3

        if body_frames:
            raised_frames = [f for f in body_frames if f.get("hand_raised")]
            obs.hand_raised = len(raised_frames) > len(body_frames) * 0.3
            if obs.hand_raised:
                sides = [f.get("hand_raise_side", "none")
                         for f in raised_frames]
                obs.hand_raise_side = max(set(sides), key=sides.count)

        obs.no_movement_sec = frames[-1].get("no_movement_sec", 0.0)

        obs.still_there_prompt = obs.no_movement_sec >= NO_MOVEMENT_TIMEOUT_SEC

        return obs


def draw_overlay(frame, frame_data: dict, obs_countdown: float):
    h, w = frame.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    green     = (0, 220, 0)
    red       = (0, 0, 220)
    yellow    = (0, 220, 220)
    white     = (255, 255, 255)
    dark      = (30, 30, 30)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 200), dark, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def txt(text, y, color=white, scale=0.55, thick=1):
        cv2.putText(frame, text, (10, y), font, scale, color, thick)

    face_col = green if frame_data.get("face_detected") else red
    txt(f"Face:      {'YES' if frame_data.get('face_detected') else 'NO'}", 25, face_col)

    gaze = frame_data.get("gaze", 0.0)
    gaze_col = green if gaze > 0.5 else yellow
    txt(f"Gaze:      {gaze:.2f}  Yaw: {frame_data.get('yaw', 0):.1f}", 50, gaze_col)

    moving_col = yellow if frame_data.get("head_moving") else white
    txt(f"Head:      {'MOVING' if frame_data.get('head_moving') else 'still'}", 75, moving_col)

    hand_col = green if frame_data.get("hand_raised") else white
    side = frame_data.get("hand_raise_side", "none")
    txt(f"Hand:      {'RAISED (' + side + ')' if frame_data.get('hand_raised') else 'down'}", 100, hand_col)

    no_mov = frame_data.get("no_movement_sec", 0.0)
    no_mov_col = red if no_mov >= NO_MOVEMENT_TIMEOUT_SEC else (
                 yellow if no_mov > 5 else white)
    txt(f"No movement: {no_mov:.1f}s", 125, no_mov_col)

    txt(f"Next obs in: {obs_countdown:.1f}s", 155, white)

    if frame_data.get("no_movement_sec", 0) >= NO_MOVEMENT_TIMEOUT_SEC:
        cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "PROMPT: Are you still there?",
                    (10, h-18), font, 0.7, white, 2)


def process_frame_data(rgb_frame, img_w: int, img_h: int,
                       tracker: MovementTracker) -> dict:
    face_detected = False
    yaw, pitch, gaze = 0.0, 0.0, 0.0
    body_detected = False
    hand_raised, hand_side = False, "none"
    left_wrist: tuple[float, float] | None = None
    right_wrist: tuple[float, float] | None = None
    face_center: tuple[float, float] | None = None
    face_size: float | None = None

    if _VISION_BACKEND == "mediapipe_tasks":
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb_frame)

        face_result = face_landmarker.detect(mp_image)
        if face_result.face_landmarks:
            face_detected = True
            landmarks = face_result.face_landmarks[0]
            yaw, pitch = estimate_head_pose(landmarks, img_w, img_h)
            gaze = estimate_gaze(yaw)

        pose_result = pose_landmarker.detect(mp_image)
        if pose_result.pose_landmarks:
            body_detected = True
            landmarks = pose_result.pose_landmarks[0]
            hand_raised, hand_side = detect_hand_raise(landmarks)

            l_w = landmarks[_PoseLandmark.LEFT_WRIST]
            if (getattr(l_w, "visibility", 1.0) or 1.0) > WRIST_VISIBILITY_MIN:
                left_wrist = (l_w.x, l_w.y)
            r_w = landmarks[_PoseLandmark.RIGHT_WRIST]
            if (getattr(r_w, "visibility", 1.0) or 1.0) > WRIST_VISIBILITY_MIN:
                right_wrist = (r_w.x, r_w.y)
    else:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        profile_faces = _detect_profile_faces(gray)
        all_faces = list(faces) + profile_faces

        face_detected = len(all_faces) > 0
        if face_detected:
            x, y, w, h = max(all_faces, key=lambda box: box[2] * box[3])
            face_box = (x, y, w, h)
            face_center = ((x + w / 2.0) / img_w, (y + h / 2.0) / img_h)
            face_size = (w * h) / float(img_w * img_h)
            gaze = 1.0 if len(faces) > 0 else 0.0
            body_detected = True
            hand_raised, hand_side = _detect_opencv_hand_raise(rgb_frame, face_box)
        else:
            gaze = 0.0

    head_moving, no_movement_sec = tracker.update(
        yaw, pitch, hand_raised, face_detected,
        left_wrist=left_wrist, right_wrist=right_wrist,
        face_center=face_center, face_size=face_size)

    return {
        "face_detected":    face_detected,
        "gaze":             gaze,
        "yaw":              yaw,
        "pitch":            pitch,
        "head_moving":      head_moving,
        "body_detected":    body_detected,
        "hand_raised":      hand_raised,
        "hand_raise_side":  hand_side,
        "no_movement_sec":  no_movement_sec,
    }


def _detect_profile_faces(gray) -> list:
    if _PROFILE_FACE_CASCADE is None or _PROFILE_FACE_CASCADE.empty():
        return []

    profile_faces = list(_PROFILE_FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    ))

    flipped = cv2.flip(gray, 1)
    flipped_faces = _PROFILE_FACE_CASCADE.detectMultiScale(
        flipped,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    width = gray.shape[1]
    for x, y, w, h in flipped_faces:
        profile_faces.append((width - x - w, y, w, h))

    return profile_faces


def _detect_opencv_hand_raise(rgb_frame, face_box) -> tuple[bool, str]:
    if face_box is None:
        return False, "none"

    img_h, img_w = rgb_frame.shape[:2]
    fx, fy, fw, fh = face_box
    face_cx = fx + fw / 2.0
    face_cy = fy + fh / 2.0

    ycrcb = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # Remove the detected face so the face itself is not counted as a hand.
    pad_x = int(fw * 0.35)
    pad_y = int(fh * 0.25)
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_y)
    x2 = min(img_w, fx + fw + pad_x)
    y2 = min(img_h, fy + fh + pad_y)
    skin_mask[y1:y2, x1:x2] = 0

    # Raised hands usually appear above or beside the head/upper torso.
    roi_bottom = min(img_h, int(face_cy + fh * 0.9))
    skin_mask[roi_bottom:, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        skin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    min_area = img_w * img_h * 0.003
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0
        if cy <= face_cy + fh * 0.4 and abs(cx - face_cx) > fw * 0.45:
            candidates.append((area, cx))

    if not candidates:
        return False, "none"

    _area, hand_cx = max(candidates, key=lambda item: item[0])
    return True, "left" if hand_cx < face_cx else "right"


def run():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check CAMERA_INDEX in config.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    tracker = MovementTracker()
    builder = ObservationBuilder()

    print("=" * 55)
    print("  MediaPipe Processor — running")
    print(f"  Observation every {OBSERVATION_INTERVAL_SEC}s")
    print(f"  'Still there?' prompt after {NO_MOVEMENT_TIMEOUT_SEC}s")
    print("  Press Q to quit")
    print("=" * 55)

    observations = []
    last_frame_data: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_data = process_frame_data(rgb, w, h, tracker)
        last_frame_data = frame_data
        builder.add_frame(frame_data)

        if builder.ready():
            obs = builder.build()
            observations.append(obs.to_dict())

            print("\n-- Observation ------------------------------------")
            print(obs.to_json())

            if obs.still_there_prompt:
                print("\n  PROMPT: 'Are you still there?'")
                tracker.reset()

        if SHOW_WINDOW:
            countdown = max(0.0, OBSERVATION_INTERVAL_SEC -
                            (time.time() - builder._last_obs_time))
            draw_overlay(frame, last_frame_data, countdown)
            cv2.imshow("Marty — MediaPipe Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()

    log_path = f"observations_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(observations, f, indent=2)

    print(f"\nSession ended. {len(observations)} observations saved to {log_path}")


if __name__ == "__main__":
    run()
