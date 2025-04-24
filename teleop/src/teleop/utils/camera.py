import cv2
import numpy as np

# Take a screenshot of the webcam and save it to a file
def take_screenshot(filename: str = "screenshot.png") -> None:
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    # Check if the frame was captured correctly
    if not ret:
        print("Error: Could not read frame.")
        return
    
    # Save the frame to a file
    cv2.imwrite(filename, frame)

    # Release the webcam
    cap.release()