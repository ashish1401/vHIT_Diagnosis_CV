import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Define smoothing function
def smooth_value(new_value, buffer):
    """Applies smoothing by averaging over recent values."""
    buffer.append(new_value)
    return np.mean(buffer)

# Define input and output paths
input_dir = "./Vhit Videos"
output_log_dir = "./output_log"
os.makedirs(output_log_dir, exist_ok=True)

# Process each video in the directory
for video_file in os.listdir(input_dir):
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize smoothing buffers
        eyeball_buffer_left = deque(maxlen=5)
        eyeball_buffer_right = deque(maxlen=5)
        state_buffer = deque(maxlen=10)  # Store last 10 frame states

        # Transition tracking
        total_transitions = []
        non_steady_start_time = None
        initial_pupil_position = None
        transition_index = 1  # Initialize transition index

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
                    right_eye_inner = 362 
                    
                    # Get coordinates
                    left_face_point = np.array([face_landmarks.landmark[left_face_edge].x * width, 
                                                face_landmarks.landmark[left_face_edge].y * height])
                    right_face_point = np.array([face_landmarks.landmark[right_face_edge].x * width, 
                                                 face_landmarks.landmark[right_face_edge].y * height])
                    left_eye_point_outer = np.array([face_landmarks.landmark[left_eye_outer].x * width, 
                                                     face_landmarks.landmark[left_eye_outer].y * height])
                    right_eye_point_inner = np.array([face_landmarks.landmark[right_eye_inner].x * width, 
                                                      face_landmarks.landmark[right_eye_inner].y * height])

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
                    left_threshold = initial_left_distance * 0.6
                    right_threshold = initial_right_distance * 0.6  

                    # Check for unsteady state
                    is_unsteady = left_distance < left_threshold or right_distance < right_threshold
                    
                    # Store state in buffer
                    state_buffer.append(is_unsteady)
                    smoothed_state = np.mean(state_buffer) > 0.5  # Majority voting in buffer

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
                            total_transitions.append({
                                "Index": transition_index,
                                "Start Time (s)": round(non_steady_start_time, 2),
                                "Time Traversed (s)": elapsed_time,
                                "Distance Traveled Left Eye (px)": left_eye_travel,
                                "Distance Traveled Right Eye (px)": right_eye_travel,
                                "Status": "Delayed" if left_eye_travel > 65 and right_eye_travel > 65 else "Normal"
                            })
                            transition_index += 1  # Increment transition index
                            
                            # Reset tracking
                            non_steady_start_time = None
                            initial_pupil_position = None
        
        cap.release()
        
        # Write log file
        log_file_path = os.path.join(output_log_dir, f"{video_file}.log")
        with open(log_file_path, "w") as log_file:
            log_file.write("Tracked Transitions:\n")
            for transition in total_transitions:
                log_file.write(str(transition) + "\n")

print("Processing complete. Logs saved in 'output_log' directory.")