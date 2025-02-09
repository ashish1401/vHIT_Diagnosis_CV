
import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture("eye_recording.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest (ROI) - adjust based on video calibration
    roi = frame[269: 795, 537: 1416]
    rows, cols, _ = roi.shape

    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # removal of noise for efficient extraction
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    
    # no binary masking 
    
    # thresholding to get the binary image
    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)

    # Use HoughCircles to detect the pupil or iris
    circles = cv2.HoughCircles(
        threshold,
        cv2.HOUGH_GRADIENT,
        dp=1.5,                # Increase dp for better circle approximation
        minDist=40,            # Minimum distance between circles
        param1=100,            # Canny high threshold
        param2=20,             # Accumulator threshold, lower to detect darker circles in white background
        minRadius=5,          # Minimum radius of the pupil or iris
        maxRadius= 50          # Maximum radius of the pupil or iris
    )

    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(roi, (i[0], i[1]), i[2], (255, 0, 0), 2)
            # Draw the center of the circle
            cv2.circle(roi, (i[0], i[1]), 2, (0, 255, 0), 3)
            break  # Stop after finding the first most likely candidate

    # Display the results
    cv2.imshow("Thresholded ROI", threshold)
    cv2.imshow("ROI with Detected Retina", roi)

    # Exit on pressing the 'Esc' key
    key = cv2.waitKey(30)
    if key == 27:  # ASCII for Esc key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
