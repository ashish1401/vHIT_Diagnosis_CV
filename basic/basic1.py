
# Description : This program detects the pupil or iris using contour detection in a video feed and draws a rectangle around it.

import cv2
import numpy as np



cap = cv2.VideoCapture("eye_recording.mp4")

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    roi = frame[269: 795, 537: 1416]
    rows, cols, _ = roi.shape

    # convert the video to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # removal of noise for efficient extraction
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    # thresholding to get the binary image
    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # utilizing bounding boxes
    for cnt in contours:    
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("Grayscale RoI", gray_roi)
    cv2.imshow("RoI", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()