import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe FaceMesh and video capture
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
cap = cv2.VideoCapture(0)

# Template storage for gaze directions
left_eye_template = {"center": None, "left": None, "right": None}
right_eye_template = {"center": None, "left": None, "right": None}

# Instruction message for template capture
print("Instructions:")
print("1. Look straight ahead and press 'c' to capture center gaze template.")
print("2. Look left and press 'l' to capture left gaze template.")
print("3. Look right and press 'r' to capture right gaze template.")
print("4. Once all templates are captured, gaze tracking will start automatically.")

# Flags to ensure all templates are captured
templates_captured = {"center": False, "left": False, "right": False}

# Variables for suspicious activity detection
suspicious_start_time = None
suspicious_duration_threshold = 2.5  # seconds for suspicious gaze duration
suspicious_display_duration = 10     # seconds to display the warning message
suspicious_display_start_time = None

# Main loop to process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert frame to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Capture key press at the start of each loop iteration
    key = cv2.waitKey(1) & 0xFF

    # Process landmarks if detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define landmarks for left and right eyes
            left_eye = np.array([
                [face_landmarks.landmark[362].x, face_landmarks.landmark[362].y],
                [face_landmarks.landmark[385].x, face_landmarks.landmark[385].y],
                [face_landmarks.landmark[387].x, face_landmarks.landmark[387].y],
                [face_landmarks.landmark[263].x, face_landmarks.landmark[263].y],
                [face_landmarks.landmark[373].x, face_landmarks.landmark[373].y],
                [face_landmarks.landmark[380].x, face_landmarks.landmark[380].y]
            ])
            right_eye = np.array([
                [face_landmarks.landmark[33].x, face_landmarks.landmark[33].y],
                [face_landmarks.landmark[160].x, face_landmarks.landmark[160].y],
                [face_landmarks.landmark[158].x, face_landmarks.landmark[158].y],
                [face_landmarks.landmark[133].x, face_landmarks.landmark[133].y],
                [face_landmarks.landmark[153].x, face_landmarks.landmark[153].y],
                [face_landmarks.landmark[144].x, face_landmarks.landmark[144].y]
            ])

            # Convert normalized coordinates to pixel coordinates
            height, width, _ = frame.shape
            left_eye = (left_eye * [width, height]).astype(int)
            right_eye = (right_eye * [width, height]).astype(int)

            # Extract bounding boxes for eye regions
            left_eye_x, left_eye_y, left_eye_w, left_eye_h = cv2.boundingRect(left_eye)
            right_eye_x, right_eye_y, right_eye_w, right_eye_h = cv2.boundingRect(right_eye)

            # Extract eye region images and convert to grayscale
            left_eye_img = cv2.cvtColor(frame[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w], cv2.COLOR_BGR2GRAY)
            right_eye_img = cv2.cvtColor(frame[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w], cv2.COLOR_BGR2GRAY)

            # Resize eye images for consistency in template matching
            target_size = (50, 50)
            left_eye_img = cv2.resize(left_eye_img, target_size)
            right_eye_img = cv2.resize(right_eye_img, target_size)

            # Capture templates based on key press
            if key == ord('c'):  # Center gaze template
                left_eye_template["center"] = left_eye_img.copy()
                right_eye_template["center"] = right_eye_img.copy()
                templates_captured["center"] = True
                print("Center gaze template captured.")
            elif key == ord('l'):  # Left gaze template
                left_eye_template["left"] = left_eye_img.copy()
                right_eye_template["left"] = right_eye_img.copy()
                templates_captured["left"] = True
                print("Left gaze template captured.")
            elif key == ord('r'):  # Right gaze template
                left_eye_template["right"] = left_eye_img.copy()
                right_eye_template["right"] = right_eye_img.copy()
                templates_captured["right"] = True
                print("Right gaze template captured.")

            # Check if all templates are captured
            if all(templates_captured.values()):
                # Function to match gaze direction using templates
                def match_gaze(eye_img, templates):
                    max_score = 0
                    gaze_direction = "Center"
                    for direction, template in templates.items():
                        if template is not None:
                            match_result = cv2.matchTemplate(eye_img, template, cv2.TM_CCOEFF_NORMED)
                            _, score, _, _ = cv2.minMaxLoc(match_result)
                            if score > max_score:
                                max_score = score
                                gaze_direction = direction
                    return gaze_direction if max_score > 0.6 else "Center"

                # Determine gaze direction for each eye
                left_gaze = match_gaze(left_eye_img, left_eye_template)
                right_gaze = match_gaze(right_eye_img, right_eye_template)

                # Determine overall gaze direction
                if left_gaze == right_gaze:
                    overall_gaze = left_gaze
                else:
                    overall_gaze = "Center"

                # Suspicious activity detection logic
                if overall_gaze in ["left", "right"]:
                    if suspicious_start_time is None:
                        suspicious_start_time = time.time()
                    elif time.time() - suspicious_start_time > suspicious_duration_threshold:
                        suspicious_display_start_time = time.time()
                else:
                    suspicious_start_time = None  # Reset if gaze is "center"

                # Display overall gaze direction
                cv2.putText(frame, f"Gaze Direction: {overall_gaze}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Display "SUSPICIOUS ACTIVITY DETECTED" if within display duration
                if suspicious_display_start_time and (time.time() - suspicious_display_start_time < suspicious_display_duration):
                    cv2.putText(frame, "SUSPICIOUS ACTIVITY DETECTED", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    suspicious_display_start_time = None  # Reset display after duration

            # Draw eye landmarks for visualization
            for point in left_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Eye/Gaze Tracking", frame)

    # Exit loop on pressing 'q'
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
