# Description : This program detects the pupil or iris in a video feed and draws a circle around it. 
# The program uses HoughCircles to detect the pupil or iris based on a binary mask created using adaptive thresholding and morphological operations. 
# The program displays the binary mask and the detected pupil or iris in the video feed. The program exits on pressing the 'Esc' key.
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

    # Apply Gaussian blur to remove noise for efficient extraction
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    # Apply binary thresholding to create a mask of dark areas (eyeball and pupil)
    _, binary_mask = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY_INV)

    # Use morphological operations to clean up small details around the pupil
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=4)

    # Apply HoughCircles to the cleaned-up mask to detect the pupil or iris
    circles = cv2.HoughCircles(
        binary_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.5,                # Increase dp for better circle approximation
        minDist=40,            # Minimum distance between circles
        param1=100,            # Canny high threshold
        param2=20,             # Accumulator threshold, lower to detect darker circles in white background
        minRadius=5,           # Minimum radius of the pupil or iris
        maxRadius=50           # Maximum radius of the pupil or iris
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
    cv2.imshow("Binary Mask", binary_mask)
    cv2.imshow("ROI with Detected Retina", roi)

    # Exit on pressing the 'Esc' key
    key = cv2.waitKey(30)
    if key == 27:  # ASCII for Esc key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
