# EyeTrac: Head and Gaze Controlled Mouse

EyeTrac is a Python application that allows you to control your mouse cursor using your head and eye movements. It uses computer vision to track your facial landmarks and gaze, providing a hands-free way to interact with your computer. The application features a learning model that continuously adapts to your movements to improve accuracy over time.

## Features

*   **Head and Gaze Tracking**: Controls the mouse cursor by tracking the orientation of your head and the direction of your gaze.
*   **Blink to Click**: Perform single and double clicks by blinking.
*   **Calibration**: A simple calibration process to adapt to your neutral head position.
*   **Adaptive Learning**: A machine learning model that learns from your movements to improve cursor accuracy and smoothness.
*   **Configurable**: All settings can be easily tweaked in a `config.json` file.
*   **Cross-Platform**: Works on Windows, macOS, and Linux.

## Demo

(Placeholder for a GIF or video demonstrating the project)

## Requirements

*   Python 3.6+
*   Webcam
*   The following Python libraries:
    *   `opencv-python`
    *   `dlib`
    *   `numpy`
    *   `pyautogui`
    *   `scikit-learn`
    *   `pandas`

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/alammahbub/eyetrac.git
    cd eyetrac
    ```
2.  Install the required Python libraries:
    ```bash
    pip install opencv-python dlib numpy pyautogui scikit-learn pandas
    ```
3.  Download the facial landmark predictor model:
    Download the `shape_predictor_68_face_landmarks.dat` file from [here](http.dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
    Extract the file and place it in the project's root directory.

## Usage

1.  Run the script from your terminal:
    ```bash
    python eye_mouse.py
    ```
2.  **Calibration**:
    *   When you first run the application, you will be prompted to calibrate.
    *   Look straight at the screen and press the 'c' key.
    *   Hold your head still for a few seconds until the calibration is complete.
    *   The calibration data is saved to `calibration.json` and will be loaded automatically the next time you run the application. You can re-calibrate at any time by pressing 'c' again.
3.  **Controlling the Cursor**:
    *   Move your head to move the cursor around the screen.
    *   Use your gaze for fine-tuned adjustments.
4.  **Clicking**:
    *   **Single Click**: Perform a single blink.
    *   **Double Click**: Perform two blinks in quick succession.
5.  **Quitting**:
    *   Press the 'q' key to quit the application.

## Configuration

The `config.json` file allows you to customize the application's settings:
```json
{
    "pose_sensitivity": 4000,
    "gaze_sensitivity": 0.5,
    "blink_threshold": 0.25,
    "blink_consecutive_frames": 2,
    "dead_zone_threshold": 0.03,
    "smoothing_factor": 0.80,
    "update_every_n_frames": 2,
    "calibration_file": "calibration.json",
    "double_blink_window": 0.5
}
```
*   `pose_sensitivity`: Controls how much the cursor moves in response to head movements.
*   `gaze_sensitivity`: Controls how much the cursor moves in response to eye movements.
*   `blink_threshold`: The eye aspect ratio threshold to detect a blink.
*   `blink_consecutive_frames`: The number of consecutive frames the eye must be closed to register a blink.
*   `dead_zone_threshold`: A threshold to prevent small, unintentional movements from moving the cursor.
*   `smoothing_factor`: Controls the smoothness of the cursor movement. A higher value results in smoother but slightly delayed movement.
*   `update_every_n_frames`: How often to update the target mouse position.
*   `calibration_file`: The name of the file to save calibration data to.
*   `double_blink_window`: The time window in seconds for detecting a double blink.

## How it Works

*   **Face and Landmark Detection**: `dlib` and `OpenCV` are used to detect the user's face and facial landmarks from the webcam feed.
*   **Head Pose Estimation**: The 3D position of the head is estimated from the landmark positions.
*   **Gaze Tracking**: The pupil is detected to estimate the direction of the gaze.
*   **Machine Learning**: A `scikit-learn` linear regression model is used to learn the mapping from head and gaze data to screen coordinates. The model is continuously retrained on new data logged in `movement_log.csv`.
*   **Mouse Control**: `pyautogui` is used to control the system's mouse cursor.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
