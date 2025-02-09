import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Load video file
video_path = "./vHIT/vHITSlowed.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Function to compute eyeball center with smoothing
def capture_eye_ball(pupil_landmarks, face_landmarks, width, height, buffer):
    points = np.array([[face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height] for i in pupil_landmarks])
    center = np.mean(points, axis=0).astype(int)
    buffer.append(center)
    smoothed_center = np.mean(buffer, axis=0).astype(int)
    return tuple(smoothed_center)

# Smoothing buffers
eyeball_buffer_left = deque(maxlen=5)
eyeball_buffer_right = deque(maxlen=5)
face_mesh_buffer = deque(maxlen=5)

def smooth_landmarks(face_landmarks, width, height, buffer):
    points = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
    buffer.append(points)
    smoothed_points = np.mean(buffer, axis=0).astype(int)
    return smoothed_points

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            
            # Draw full face mesh
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Define facial and eye landmarks
            left_face_edge = 234  # Leftmost face landmark
            right_face_edge = 454  # Rightmost face landmark
            left_eye_outer = 133  # Rightmost part of left eye
            right_eye_outer = 362  # Leftmost part of right eye
            
            # Get coordinates
            left_face_point = np.array([face_landmarks.landmark[left_face_edge].x * width, face_landmarks.landmark[left_face_edge].y * height])
            right_face_point = np.array([face_landmarks.landmark[right_face_edge].x * width, face_landmarks.landmark[right_face_edge].y * height])
            left_eye_point = np.array([face_landmarks.landmark[left_eye_outer].x * width, face_landmarks.landmark[left_eye_outer].y * height])
            right_eye_point = np.array([face_landmarks.landmark[right_eye_outer].x * width, face_landmarks.landmark[right_eye_outer].y * height])
            
            # Draw key points used for distance calculation in thick red
            cv2.circle(frame, tuple(left_face_point.astype(int)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_face_point.astype(int)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(left_eye_point.astype(int)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_eye_point.astype(int)), 6, (0, 0, 255), -1)
            
            # Compute distances
            left_distance = np.linalg.norm(left_face_point - left_eye_point)
            right_distance = np.linalg.norm(right_face_point - right_eye_point)
            
            # Store reference distances on first frame
            if 'initial_left_distance' not in locals():
                initial_left_distance = left_distance
                initial_right_distance = right_distance
            
            # Check for unsteady state
            if left_distance < (initial_left_distance / 2) or right_distance < (initial_right_distance / 2):
                cv2.putText(frame, "Not Steady", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Steady State", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out.write(frame)
    cv2.imshow("Face and Eye Stability Tracking", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
