import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Get screen size
screen_w, screen_h = pyautogui.size()

while True:
    # Capture frame
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip for natural movement
    frame_h, frame_w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    # Get face landmarks
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Extract right and left iris landmarks
        right_iris = [landmarks[468], landmarks[469], landmarks[470], landmarks[471]]
        left_iris = [landmarks[472], landmarks[473], landmarks[474], landmarks[475]]

        # Calculate center of right iris
        right_x = int(np.mean([p.x for p in right_iris]) * frame_w)
        right_y = int(np.mean([p.y for p in right_iris]) * frame_h)
        cv2.circle(frame, (right_x, right_y), 5, (255, 0, 0), -1)  # Blue circle for right iris

        # Calculate center of left iris
        left_x = int(np.mean([p.x for p in left_iris]) * frame_w)
        left_y = int(np.mean([p.y for p in left_iris]) * frame_h)
        cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)  # Green circle for left iris

        # Use the average of both eyes for stability
        avg_x = (right_x + left_x) // 2
        avg_y = (right_y + left_y) // 2

        # Map pupil position to screen coordinates
        screen_x = np.interp(avg_x, [0, frame_w], [0, screen_w])
        screen_y = np.interp(avg_y, [0, frame_h], [0, screen_h])

        # Move cursor smoothly
        pyautogui.moveTo(screen_x, screen_y, duration=0.05)

    # Show frame with tracking
    cv2.imshow('Eye-Controlled Mouse', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
