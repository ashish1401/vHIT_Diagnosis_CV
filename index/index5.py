# Medium 2 - Eye/Gaze Tracking L.R.C Template Matching - Haar Cascade

# Two staged Haar Cascade based eye tracking with template matching for gaze detection
# The program uses Haar cascades for face and eye detection and template matching for gaze detection.
# The program captures templates for left, center, and right gaze directions and matches them with the current eye region to determine the gaze direction.
# The program allows the user to capture templates for each gaze direction by pressing 'L', 'C', and 'R' keys.
# The program then performs template matching to determine the gaze direction based on the best match with the captured templates.
import cv2
import numpy as np

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for storing templates for each gaze direction
templates = {'left': None, 'center': None, 'right': None}
template_matching_threshold = 0.8  # Threshold for template matching-


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
                cv2.circle(eye_region, pupil_center, 3, (255, 0, 0), -1)

    return pupil_center, thresholded_eye

def capture_template(eye_region, label):
    """Capture a template for a specific gaze direction."""
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.equalizeHist(gray_eye)
    templates[label] = gray_eye
    print(f"Template for '{label}' captured.")

def match_template(eye_region):
    """Match the current eye region with templates to determine gaze direction."""
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.equalizeHist(gray_eye)

    best_match = None
    best_score = 0

    for label, template in templates.items():
        if template is not None:
            res = cv2.matchTemplate(gray_eye, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score and max_val >= template_matching_threshold:
                best_score = max_val
                best_match = label

    return best_match

setup_complete = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        face_region = frame[fy:fy + fh, fx:fx + fw]
        face_gray = gray[fy:fy + fh, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        
        for (ex, ey, ew, eh) in eyes[:2]:  # Track only the first two eyes detected
            cv2.rectangle(face_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            eye_region = face_region[ey:ey + eh, ex:ex + ew]
            pupil_center, thresholded_eye = detect_pupil(eye_region)

            if not setup_complete:
                # Setup phase to capture templates
                cv2.putText(frame, "Press L for Left, C for Center, R for Right to capture templates",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                key = cv2.waitKey(1)
                if key == ord('l'):
                    capture_template(eye_region, 'left')
                elif key == ord('c'):
                    capture_template(eye_region, 'center')
                elif key == ord('r'):
                    capture_template(eye_region, 'right')
                
                # Check if all templates have been captured
                if templates['left'] is not None and templates['center'] is not None and templates['right'] is not None:
                    setup_complete = True
                    print("Template setup complete.")
            else:
                # Perform template matching to determine gaze direction
                gaze_direction = match_template(eye_region)
                if gaze_direction:
                    cv2.putText(frame, f"Gaze: {gaze_direction}", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow("Thresholded Eye", thresholded_eye)

    cv2.imshow("Eye/Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
