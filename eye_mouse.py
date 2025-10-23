import cv2
import dlib
import numpy as np
import pyautogui
import json
import os
import time
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import pandas as pd
except ImportError:
    print("Scikit-learn or pandas not found. Please install them using: pip install scikit-learn pandas")
    exit()

# --- Configuration Loading ---
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()

# --- Constants from Config ---
POSE_SENSITIVITY = config["pose_sensitivity"]
GAZE_SENSITIVITY = config["gaze_sensitivity"]
BLINK_THRESHOLD = config["blink_threshold"]
BLINK_CONSECUTIVE_FRAMES = config["blink_consecutive_frames"]
DEAD_ZONE_THRESHOLD = config["dead_zone_threshold"]
SMOOTHING_FACTOR = config["smoothing_factor"]
UPDATE_EVERY_N_FRAMES = config["update_every_n_frames"]
CALIBRATION_FILE = config["calibration_file"]
LOG_FILE = "movement_log.csv"
RETRAIN_EVERY_N_FRAMES = 100
DOUBLE_BLINK_WINDOW = config["double_blink_window"]
LOGGING_MOVEMENT_THRESHOLD = config["logging_movement_threshold"]
LOGGING_INPUT_THRESHOLD = config["logging_input_threshold"]

# --- Initialization ---
pyautogui.FAILSAFE = False
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
except RuntimeError:
    print("Error: 'shape_predictor_68_face_landmarks.dat' not found.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

screen_width, screen_height = pyautogui.size()

# --- Helper Functions ---
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0: return 0.3
    return (A + B) / (2.0 * C)

def get_pupil_center(gray_frame, eye_landmarks):
    try:
        eye_region = cv2.boundingRect(eye_landmarks)
        x, y, w, h = eye_region
        padding = 10
        x, y, w, h = x - padding, y - padding, w + 2*padding, h + 2*padding
        x, y = max(x, 0), max(y, 0)
        eye_roi = gray_frame[y:y+h, x:x+w]
        if eye_roi.size == 0: return None

        _, threshold_eye = cv2.threshold(eye_roi, 55, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        threshold_eye = cv2.erode(threshold_eye, kernel, iterations=2)
        threshold_eye = cv2.dilate(threshold_eye, kernel, iterations=4)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (x + cx, y + cy)
    except Exception:
        return None
    return None

def save_calibration(h_ratio, v_ratio):
    data = {"neutral_horizontal_ratio": h_ratio, "neutral_vertical_ratio": v_ratio}
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f)
    print(f"Calibration data saved to {CALIBRATION_FILE}")

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            try:
                data = json.load(f)
                print(f"Calibration data loaded from {CALIBRATION_FILE}")
                return data["neutral_horizontal_ratio"], data["neutral_vertical_ratio"]
            except (json.JSONDecodeError, KeyError):
                print(f"Error reading {CALIBRATION_FILE}. Please re-calibrate.")
    return None, None

# --- Selective Logging ---
class SelectiveLogger:
    def __init__(self, log_file, movement_threshold, input_threshold):
        self.log_file = log_file
        self.movement_threshold = movement_threshold
        self.input_threshold = input_threshold
        self.last_logged_h_ratio = -1
        self.last_logged_v_ratio = -1
        self.last_logged_gaze_x = -1
        self.last_logged_gaze_y = -1
        self.last_logged_final_x = -1
        self.last_logged_final_y = -1

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("timestamp,h_ratio,v_ratio,gaze_x,gaze_y,target_x,target_y,final_x,final_y\n")

    def log(self, timestamp, h_ratio, v_ratio, gaze_x, gaze_y, target_x, target_y, final_x, final_y):
        h_ratio_diff = abs(h_ratio - self.last_logged_h_ratio)
        v_ratio_diff = abs(v_ratio - self.last_logged_v_ratio)
        gaze_x_diff = abs(gaze_x - self.last_logged_gaze_x)
        gaze_y_diff = abs(gaze_y - self.last_logged_gaze_y)
        final_x_diff = abs(final_x - self.last_logged_final_x)
        final_y_diff = abs(final_y - self.last_logged_final_y)

        input_changed = (h_ratio_diff > self.input_threshold or
                         v_ratio_diff > self.input_threshold or
                         gaze_x_diff > self.input_threshold or
                         gaze_y_diff > self.input_threshold)

        output_changed = (final_x_diff > self.movement_threshold or
                          final_y_diff > self.movement_threshold)

        if input_changed or output_changed:
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp},{h_ratio},{v_ratio},{gaze_x},{gaze_y},{target_x},{target_y},{final_x},{final_y}\n")

            self.last_logged_h_ratio = h_ratio
            self.last_logged_v_ratio = v_ratio
            self.last_logged_gaze_x = gaze_x
            self.last_logged_gaze_y = gaze_y
            self.last_logged_final_x = final_x
            self.last_logged_final_y = final_y

logger = SelectiveLogger(LOG_FILE, LOGGING_MOVEMENT_THRESHOLD, LOGGING_INPUT_THRESHOLD)

# --- Learning Model ---
class MovementModel:
    def __init__(self):
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.trained = False

    def train(self):
        if not os.path.exists(LOG_FILE):
            return

        data = pd.read_csv(LOG_FILE)
        if len(data) < 10: # Don't train with too little data
            return

        X = data[['h_ratio', 'v_ratio', 'gaze_x', 'gaze_y']]
        y_x = data['final_x']
        y_y = data['final_y']

        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)
        self.trained = True
        print("Model retrained.")

    def predict(self, h_ratio, v_ratio, gaze_x, gaze_y):
        if self.trained:
            features = np.array([h_ratio, v_ratio, gaze_x, gaze_y]).reshape(1, -1)
            pred_x = self.model_x.predict(features)
            pred_y = self.model_y.predict(features)
            return pred_x[0], pred_y[0]
        else:
            # Fallback to the original calculation if not trained
            dx_ratio = h_ratio - neutral_horizontal_ratio
            dy_ratio = v_ratio - neutral_vertical_ratio

            if abs(dx_ratio) < DEAD_ZONE_THRESHOLD: dx_ratio = 0
            if abs(dy_ratio) < DEAD_ZONE_THRESHOLD: dy_ratio = 0

            target_x = screen_width / 2 + dx_ratio * POSE_SENSITIVITY
            target_y = screen_height / 2 + dy_ratio * POSE_SENSITIVITY * 2
            return target_x, target_y

model = MovementModel()
model.train() # Initial training

# --- State Variables ---
neutral_horizontal_ratio, neutral_vertical_ratio = load_calibration()
calibration_frames = 30
calibration_counter = 0
calibration_data_hor, calibration_data_ver = [], []
calibrating = False
blink_counter = 0
frame_counter = 0
retrain_counter = 0
smoothed_mouse_x, smoothed_mouse_y = screen_width / 2, screen_height / 2
_target_mouse_x, _target_mouse_y = screen_width / 2, screen_height / 2
single_blink_detected = False
last_blink_time = 0

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_counter += 1
    retrain_counter += 1
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    face_detected = len(faces) > 0

    if face_detected:
        for face in faces:
            landmarks = predictor(gray, face)

            # --- Head Pose Estimation (Primary Control) ---
            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            left_cheek = (landmarks.part(0).x, landmarks.part(0).y)
            right_cheek = (landmarks.part(16).x, landmarks.part(16).y)
            nasion = (landmarks.part(27).x, landmarks.part(27).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)

            epsilon = 1e-6
            horizontal_ratio = np.linalg.norm(np.array(nose_tip) - np.array(left_cheek)) / (np.linalg.norm(np.array(nose_tip) - np.array(right_cheek)) + epsilon)
            vertical_ratio = np.linalg.norm(np.array(nose_tip) - np.array(nasion)) / (np.linalg.norm(np.array(nose_tip) - np.array(chin)) + epsilon)

            # --- Calibration ---
            if calibrating:
                if calibration_counter < calibration_frames:
                    calibration_data_hor.append(horizontal_ratio)
                    calibration_data_ver.append(vertical_ratio)
                    cv2.putText(frame, f"Calibrating... {calibration_counter}/{calibration_frames}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    calibration_counter += 1
                else:
                    if calibration_data_hor and calibration_data_ver:
                        neutral_horizontal_ratio = np.mean(calibration_data_hor)
                        neutral_vertical_ratio = np.mean(calibration_data_ver)
                        save_calibration(neutral_horizontal_ratio, neutral_vertical_ratio)
                    calibrating = False
                    calibration_data_hor, calibration_data_ver = [], []

            # --- Cursor Control ---
            if neutral_horizontal_ratio is not None and neutral_vertical_ratio is not None:
                # --- Gaze Refinement (Secondary Control) ---
                right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                pupil_center = get_pupil_center(gray, right_eye_landmarks)
                gaze_offset_x, gaze_offset_y = 0, 0
                if pupil_center is not None:
                    eye_center = right_eye_landmarks.mean(axis=0).astype(int)
                    gaze_offset_x = pupil_center[0] - eye_center[0]
                    gaze_offset_y = pupil_center[1] - eye_center[1]

                if frame_counter % UPDATE_EVERY_N_FRAMES == 0:
                    _target_mouse_x, _target_mouse_y = model.predict(horizontal_ratio, vertical_ratio, gaze_offset_x, gaze_offset_y)

                smoothed_mouse_x = (SMOOTHING_FACTOR * smoothed_mouse_x) + ((1 - SMOOTHING_FACTOR) * _target_mouse_x)
                smoothed_mouse_y = (SMOOTHING_FACTOR * smoothed_mouse_y) + ((1 - SMOOTHING_FACTOR) * _target_mouse_y)

                final_x = smoothed_mouse_x
                final_y = smoothed_mouse_y

                final_x = max(0, min(screen_width - 1, final_x))
                final_y = max(0, min(screen_height - 1, final_y))

                pyautogui.moveTo(final_x, final_y)

                # --- Logging and Learning ---
                timestamp = time.time()
                logger.log(timestamp, horizontal_ratio, vertical_ratio, gaze_offset_x, gaze_offset_y, _target_mouse_x, _target_mouse_y, final_x, final_y)
                if retrain_counter >= RETRAIN_EVERY_N_FRAMES:
                    model.train()
                    retrain_counter = 0


            # --- Blink Detection ---
            left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            ear = (eye_aspect_ratio(left_eye_landmarks) + eye_aspect_ratio(right_eye_landmarks)) / 2.0

            if ear < BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    if single_blink_detected and (current_time - last_blink_time) < DOUBLE_BLINK_WINDOW:
                        pyautogui.doubleClick()
                        print("Double blink detected - Double Click!")
                        single_blink_detected = False
                    else:
                        single_blink_detected = True
                        last_blink_time = current_time
                blink_counter = 0

            # --- Visualization ---
            # Draw face boundary points
            for i in range(0, 17):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # --- Single Click Handling ---
    if single_blink_detected and (time.time() - last_blink_time) >= DOUBLE_BLINK_WINDOW:
        pyautogui.click()
        print("Single blink detected - Click!")
        single_blink_detected = False

    # --- UI Instructions ---
    if neutral_horizontal_ratio is None: cv2.putText(frame, "Press 'c' to Calibrate", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else: cv2.putText(frame, "Press 'c' to Re-Calibrate", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "CLICK: BLINK | DBL-CLICK: DBL-BLINK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to Quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Head and Gaze Mouse', frame)

    # --- Key Handling ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'):
        print("Starting calibration...")
        calibrating = True
        calibration_counter = 0
        neutral_horizontal_ratio, neutral_vertical_ratio = None, None
        calibration_data_hor, calibration_data_ver = [], []

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
