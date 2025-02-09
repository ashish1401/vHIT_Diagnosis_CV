import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Function to calculate eye aspect ratio (for blink detection)
def calculate_eye_aspect_ratio(eye_points):
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Function to find the eyeball (pupil) center using depth
def calculate_pupil_position(eye_points, eye_depths):
    # Find landmark with lowest Z value (closest to camera)
    pupil_index = np.argmin(eye_depths)
    return tuple(eye_points[pupil_index])

# Load video file
video_path = "./vHIT/vHITSlowed.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer
out = cv2.VideoWriter('./output/output_slowed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define eye landmarks (separate upper/lower eyelids and eyeball)
            left_eye = [362, 385, 387, 263, 373, 380]  # Left eye boundary
            right_eye = [33, 160, 158, 133, 153, 144]  # Right eye boundary
            left_pupil = [474, 475, 476, 477]  # Left eyeball center
            right_pupil = [469, 470, 471, 472]  # Right eyeball center

            height, width, _ = frame.shape

            # Convert normalized coordinates to absolute pixel values
            left_eye_points = np.array([[face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height] for i in left_eye], dtype=int)
            right_eye_points = np.array([[face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height] for i in right_eye], dtype=int)

            left_pupil_points = np.array([[face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height] for i in left_pupil], dtype=int)
            right_pupil_points = np.array([[face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height] for i in right_pupil], dtype=int)

            left_eye_depths = np.array([face_landmarks.landmark[i].z for i in left_pupil])
            right_eye_depths = np.array([face_landmarks.landmark[i].z for i in right_pupil])

            # Compute pupil center
            left_pupil_center = calculate_pupil_position(left_pupil_points, left_eye_depths)
            right_pupil_center = calculate_pupil_position(right_pupil_points, right_eye_depths)

            # Calculate EAR (blink detection)
            left_ear = calculate_eye_aspect_ratio(left_eye_points)
            right_ear = calculate_eye_aspect_ratio(right_eye_points)

            # Draw eye landmarks
            for point in left_eye_points:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            for point in right_eye_points:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

            # Draw pupil center (tracks eyeball motion)
            cv2.circle(frame, left_pupil_center, 4, (0, 0, 255), -1)
            cv2.circle(frame, right_pupil_center, 4, (0, 0, 255), -1)

            # Blink detection threshold
            ear_threshold = 0.25
            if left_ear < ear_threshold and right_ear < ear_threshold:
                cv2.putText(frame, "Eyes Closed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Eyes Open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Track gaze direction
            if left_pupil_center[0] < left_eye_points[3][0]:  # Leftward gaze
                cv2.putText(frame, "Looking Left", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif left_pupil_center[0] > left_eye_points[0][0]:  # Rightward gaze
                cv2.putText(frame, "Looking Right", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Eye/Gaze Tracking", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
