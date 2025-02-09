import cv2
import mediapipe as mp
import numpy as np

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

# Function to compute eyeball center
def capture_eye_ball(pupil_landmarks, face_landmarks, width, height):
    points = np.array([[face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height] for i in pupil_landmarks])
    center = np.mean(points, axis=0).astype(int)
    return tuple(center)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Create a black frame of the same size as the input video
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            
            # Define pupil landmarks
            left_pupil = [474, 475, 476, 477]
            right_pupil = [469, 470, 471, 472]
            
            # Calculate pupil centers
            left_pupil_center = capture_eye_ball(left_pupil, face_landmarks, width, height)
            right_pupil_center = capture_eye_ball(right_pupil, face_landmarks, width, height)
            
            # Draw full face mesh
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)       
            
            # Draw eye mesh
            eye_landmarks = [362, 385, 387, 263, 373, 380, 33, 160, 158, 133, 153, 144]
            for i in eye_landmarks:
                landmark = face_landmarks.landmark[i]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # Draw pupil centers
            cv2.circle(frame, left_pupil_center, 4, (255, 0, 0), -1)
            cv2.circle(frame, right_pupil_center, 4, (255, 0, 0), -1)
            
            # Detect steady state by checking minimal movement
            if np.linalg.norm(np.array(left_pupil_center) - np.array(right_pupil_center)) < 5:
                cv2.putText(frame, "Steady State", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Steady", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    out.write(frame)
    cv2.imshow("Eye and Face Mesh Tracking", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
