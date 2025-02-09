import cv2
import numpy as np
from collections import deque

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Calibration variable for the center position
calibration_data = {'center': None}
calibration_complete = False

# Smoothing and persistence variables
pupil_positions = deque(maxlen=5)  # Stores last 5 pupil positions for smoothing
gaze_history = deque(maxlen=10)    # Stores recent gaze directions
gaze_threshold = 6  # Minimum frames in a row with the same gaze direction to confirm change
threshold_offset = 0.25  # Adjusted threshold offset for left and right gaze

def detect_pupil(eye_region):
    """Detects the pupil in the eye region."""
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.equalizeHist(gray_eye)
    thresholded_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_center = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 20:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pupil_center = (cx, cy)
                cv2.circle(eye_region, pupil_center, 3, (255, 0, 0), -1)  # Blue dot on the pupil

    return pupil_center, thresholded_eye

def calibrate_gaze(pupil_position, ew):
    """Function to calibrate the center gaze position."""
    global calibration_data, calibration_complete
    key = cv2.waitKey(1)
    if key == ord('c'):  # Calibrate "Center" position
        calibration_data['center'] = pupil_position[0] / ew  # Normalize to eye width
        calibration_complete = True
        print("Center calibrated at:", calibration_data['center'])

def get_smoothed_pupil_position():
    """Calculate the moving average of recent pupil positions."""
    if len(pupil_positions) == 0:
        return None
    avg_position = np.mean(pupil_positions, axis=0)
    return avg_position

def determine_gaze_direction(px, ew):
    """Determine gaze direction based on smoothed pupil position and calibrated center."""
    normalized_px = px / ew  # Normalize x position to eye width
    center_pos = calibration_data['center']
    
    # Debugging: print the normalized position
    print(f"Normalized Position: {normalized_px:.2f}, Center Position: {center_pos:.2f}")

    # Adjusted logic for determining gaze direction
    if normalized_px < center_pos - threshold_offset:
        return "Looking Left"
    elif normalized_px > center_pos + threshold_offset:
        return "Looking Right"
    else:
        return "Looking Center"

def get_stable_gaze(gaze_direction):
    """Determine a stable gaze direction based on recent history."""
    gaze_history.append(gaze_direction)
    # Check if we have enough consistent gaze directions in history
    if gaze_history.count(gaze_direction) >= gaze_threshold:
        return gaze_direction
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (fx, fy, fw, fh) in faces:
        # Draw face bounding box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # Extract the face region
        face_region = frame[fy:fy + fh, fx:fx + fw]
        face_gray = gray[fy:fy + fh, fx:fx + fw]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        
        for (ex, ey, ew, eh) in eyes[:2]:  # Track only the first two eyes detected
            # Draw eye bounding box
            cv2.rectangle(face_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

            # Extract the eye region from the face
            eye_region = face_region[ey:ey + eh, ex:ex + ew]

            # Detect the pupil
            pupil_center, thresholded_eye = detect_pupil(eye_region)
            if pupil_center:
                # Smooth pupil position with a moving average
                pupil_positions.append(pupil_center[0])
                smoothed_pupil_position = get_smoothed_pupil_position()
                
                if not calibration_complete:
                    # Perform calibration if not complete
                    calibrate_gaze(pupil_center, ew)
                    cv2.putText(frame, "Press C for Center calibration",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif smoothed_pupil_position is not None:
                    # Get gaze direction based on calibrated center and smoothed position
                    gaze_direction = determine_gaze_direction(smoothed_pupil_position, ew)
                    stable_gaze = get_stable_gaze(gaze_direction)
                    if stable_gaze:
                        cv2.putText(frame, stable_gaze, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
            # Display thresholded eye for debugging purposes
            cv2.imshow("Thresholded Eye", thresholded_eye)

    # Display the frame with gaze tracking
    cv2.imshow("Eye/Gaze Tracking", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
