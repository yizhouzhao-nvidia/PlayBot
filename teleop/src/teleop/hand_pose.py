import cv2
import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Target frame rate
target_fps = 20
frame_interval = 1.0 / target_fps


if __name__ == "__main__":

    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    # check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.2) as hands:
            
            while True:
                start_time = time.time()
                
                # Read a frame from the webcam
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                # Display the resulting frame
                cv2.imshow('Webcam (20 Hz)', frame)
                
                results = hands.process(frame)

                # Print handedness and draw hand landmarks on the image.
                
                print('Handedness:', results.multi_handedness)
                # if not results.multi_hand_landmarks:
                #     continue
                image_height, image_width, _ = frame.shape
                # annotated_image = frame.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        print('hand_landmarks:', hand_landmarks)
                        print(
                            f'Index finger tip coordinates: (',
                            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                        )
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                
                # Display the annotated image
                cv2.imshow('Annotated Image', frame)

                # Calculate elapsed time and wait to maintain 20Hz
                elapsed_time = time.time() - start_time
                wait_time = max(1, int((frame_interval - elapsed_time) * 1000))  # in milliseconds

                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break

    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()