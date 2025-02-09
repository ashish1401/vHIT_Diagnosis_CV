import cv2
import numpy as np

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

def detect_pupil(eye_region):
    # Convert to grayscale and enhance contrast
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.equalizeHist(gray_eye)
    
    # Apply adaptive thresholding to handle lighting variations
    thresholded_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_center = None

    if contours:
        # Find the largest contour, assuming it's the pupil, and filter by size
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 20:  # Adjust size threshold as needed
            # Calculate moments to find the center of the pupil
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pupil_center = (cx, cy)
                # Draw the center of the pupil
                cv2.circle(eye_region, pupil_center, 3, (255, 0, 0), -1)  # Blue dot on the pupil

    return pupil_center, thresholded_eye

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
                # Calculate relative position of the pupil within the eye
                px, py = pupil_center
                if px < ew // 3:
                    gaze_direction = "Looking Left"
                elif px > 2 * ew // 3:
                    gaze_direction = "Looking Right"
                else:
                    gaze_direction = "Looking Center"
                
                # Display the gaze direction
                cv2.putText(frame, gaze_direction, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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
