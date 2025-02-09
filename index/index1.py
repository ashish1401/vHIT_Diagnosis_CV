
# Basic 2 - Deep Learning Based - MediaPipe FaceMesh 
# The MediaPipe FaceMesh model is a deep learning-based model that can detect facial landmarks in real-time.
# The model can detect 468 facial landmarks, including the eyes, nose, mouth, and other facial features.
# We can use the FaceMesh model to detect eye landmarks and calculate the eye aspect ratio (EAR) to determine if the eyes are open or closed.
# The EAR is a measure of eye openness and can be used to detect drowsiness or fatigue in drivers or individuals.

import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe FaceMesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the eye aspect ratio
def calculate_eye_aspect_ratio(eye_points):
    # Calculate the distances between the two vertical eye landmarks
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Calculate the distance between the horizontal eye landmarks
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    # Calculate the EAR
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Function to calculate the eye center (approximate position of the pupil/retina)
def calculate_eye_center(eye_points):
    # Calculate the centroid of the eye landmarks
    cx = int(np.mean(eye_points[:, 0]))
    cy = int(np.mean(eye_points[:, 1]))
    return (cx, cy)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Loop to read each frame from the webcam
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get face landmarks
    results = face_mesh.process(rgb_frame)

    # If face landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates for left and right eyes
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

            # Convert normalized coordinates to pixel values
            height, width, _ = frame.shape
            left_eye = (left_eye * [width, height]).astype(int)
            right_eye = (right_eye * [width, height]).astype(int)

            # Calculate EAR for both eyes
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)

            # Calculate eye centers (retina/pupil approximation)
            left_eye_center = calculate_eye_center(left_eye)
            right_eye_center = calculate_eye_center(right_eye)

            # Draw eye landmarks
            for point in left_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

            # Draw blue dot at the eye centers to represent the retina/pupil
            cv2.circle(frame, left_eye_center, 3, (255, 0, 0), -1)
            cv2.circle(frame, right_eye_center, 3, (255, 0, 0), -1)

            # Check if eyes are open or closed
            ear_threshold = 0.25  # Threshold for detecting closed eyes
            if left_ear < ear_threshold and right_ear < ear_threshold:
                cv2.putText(frame, "Eyes Closed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Eyes Open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Eye/Gaze Tracking", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
