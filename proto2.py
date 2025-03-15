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

# Smoothing buffers
eyeball_buffer_left = deque(maxlen=5)
eyeball_buffer_right = deque(maxlen=5)
face_mesh_buffer = deque(maxlen=5)
state_buffer = deque(maxlen=10)  # Store last 10 frame states

# Transition tracking
total_transitions = []
non_steady_start_time = None
initial_pupil_position = None
transition_index = 1  # Initialize transition index
last_transition_data = None  # Stores last transition data to keep displaying it until the next unsteady state

def smooth_value(new_value, buffer):
    """Applies smoothing by averaging over recent values."""
    buffer.append(new_value)
    return np.mean(buffer)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get time in seconds
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            
            # Define facial and eye landmarks
            left_face_edge = 234  # Leftmost face landmark
            right_face_edge = 454  # Rightmost face landmark
            left_eye_outer = 33  
            left_eye_inner = 133
            right_eye_outer = 263
            right_eye_inner = 362 
            left_pupil = [474, 475, 476, 477]  # Left eyeball center
            right_pupil = [469, 470, 471, 472]  # Right eyeball center
            
            # Get coordinates
            left_face_point = np.array([face_landmarks.landmark[left_face_edge].x * width, 
                                        face_landmarks.landmark[left_face_edge].y * height])
            right_face_point = np.array([face_landmarks.landmark[right_face_edge].x * width, 
                                         face_landmarks.landmark[right_face_edge].y * height])
            left_eye_point_outer = np.array([face_landmarks.landmark[left_eye_outer].x * width, 
                                             face_landmarks.landmark[left_eye_outer].y * height])
            right_eye_point_inner = np.array([face_landmarks.landmark[right_eye_inner].x * width, 
                                              face_landmarks.landmark[right_eye_inner].y * height])
            
            # Draw key points used for distance calculation
            cv2.circle(frame, tuple(left_face_point.astype(int)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_face_point.astype(int)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(left_eye_point_outer.astype(int)), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_eye_point_inner.astype(int)), 6, (0, 255, 255), -1)
            
            # Draw pupil markers
            for i in left_pupil + right_pupil:
                landmark = face_landmarks.landmark[i]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Thin blue marker for eyeball

            # Compute distances
            left_distance = np.linalg.norm(left_face_point - left_eye_point_outer)
            right_distance = np.linalg.norm(right_face_point - right_eye_point_inner)

            # Apply smoothing
            left_distance = smooth_value(left_distance, eyeball_buffer_left)
            right_distance = smooth_value(right_distance, eyeball_buffer_right)

            # Store reference distances on first frame
            if 'initial_left_distance' not in locals():
                initial_left_distance = left_distance
                initial_right_distance = right_distance

            # Compute thresholds dynamically
            left_threshold = initial_left_distance * 0.6  # Adjusted based on average
            right_threshold = initial_right_distance * 0.6  

            # Check for unsteady state
            is_unsteady = left_distance < left_threshold or right_distance < right_threshold
            
            # Store state in buffer
            state_buffer.append(is_unsteady)
            smoothed_state = np.mean(state_buffer) > 0.5  # Majority voting in buffer

            # Display state on the frame
            state_text = "Steady State" if not smoothed_state else "Not Steady"
            state_color = (0, 255, 0) if not smoothed_state else (0, 0, 255)
            cv2.putText(frame, state_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)

            # Detect transitions
            if smoothed_state:  # Not Steady
                if non_steady_start_time is None:
                    non_steady_start_time = current_time  # Start timing
                    initial_pupil_position = (left_eye_point_outer, right_eye_point_inner)  # Store initial position
            else:  # Steady
                if non_steady_start_time is not None:
                    elapsed_time = round(current_time - non_steady_start_time, 2)
                    
                    # Compute total movement distance
                    left_eye_travel = round(np.linalg.norm(initial_pupil_position[0] - left_eye_point_outer), 2)
                    right_eye_travel = round(np.linalg.norm(initial_pupil_position[1] - right_eye_point_inner), 2)
                    
                    # Store transition data
                    last_transition_data = {
                        "Index": transition_index,
                        "Start Time (s)": round(non_steady_start_time, 2),
                        "Time Traversed (s)": elapsed_time,
                        "Distance Traveled Left Eye (px)": left_eye_travel,
                        "Distance Traveled Right Eye (px)": right_eye_travel,
                        "Status": "Delayed" if left_eye_travel > 65 and right_eye_travel > 65 else "Normal"
                    }
                    total_transitions.append(last_transition_data)
                    transition_index += 1  # Increment transition index
                    
                    # Reset tracking
                    non_steady_start_time = None
                    initial_pupil_position = None
            
            # Keep displaying the last transition data at bottom center until the next unsteady state
            if last_transition_data:
                text = f"Index:{last_transition_data['Index']} | L: {last_transition_data['Distance Traveled Left Eye (px)']}px | R: {last_transition_data['Distance Traveled Right Eye (px)']}px | {last_transition_data['Status']}"
                cv2.putText(frame, text, (frame_width // 4, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Face and Eye Stability Tracking", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
