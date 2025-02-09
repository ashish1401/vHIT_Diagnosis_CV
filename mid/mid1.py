# Basic 1 - Eye Tracking - Haar Cascade 
# Description: Detects faces and eyes in a live video feed from the webcam. The program focuses on the eye region to detect squinting eyes and track eye movement using contour detection based on adaptive thresholding. 
# The program uses Haar cascades for face and eye detection and applies Gaussian blur and adaptive thresholding to focus on the pupil for eye movement tracking. 
# The program displays the thresholded image of the eye and draws rectangles around the detected eyes in the live video feed. The program exits on pressing the 'Esc' key.

import cv2
import numpy as np

# Load Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for better detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Focus on the face region for eye detection
        face_roi_gray = gray_frame[y:y + h, x:x + w]
        face_roi_color = frame[y:y + h, x:x + w]

        # Adjust minNeighbors to lower value for detecting squinting eyes
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3)

        for (ex, ey, ew, eh) in eyes:
            # Enlarge eye region slightly to account for squinting
            ex_start, ey_start = max(0, ex - 5), max(0, ey - 5)
            ex_end, ey_end = ex + ew + 5, ey + eh + 5
            eye_roi_gray = face_roi_gray[ey_start:ey_end, ex_start:ex_end]
            eye_roi_color = face_roi_color[ey_start:ey_end, ex_start:ex_end]

            # Apply Gaussian blur and adaptive threshold to focus on the pupil
            blurred_eye = cv2.GaussianBlur(eye_roi_gray, (7, 7), 0)
            threshold_eye = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Draw rectangles around detected eyes
            cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Display thresholded image of the eye
            cv2.imshow("Thresholded Eye", threshold_eye)

            # Contour detection for eye movement tracking
            contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for cnt in contours:
                (cx, cy, cw, ch) = cv2.boundingRect(cnt)
                cv2.rectangle(eye_roi_color, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 1)
                break  # Only draw the largest contour

    # Show the resulting frame with detections
    cv2.imshow("Live Eye Detection", frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
