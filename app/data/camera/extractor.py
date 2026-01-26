import cv2
import numpy as np

# Lazy MediaPipe imports
_mp_holistic = None
_mp_drawing = None

def _get_mp():
    global _mp_holistic, _mp_drawing
    if _mp_holistic is None:
        import mediapipe as mp
        try:
            from mediapipe.python.solutions import holistic as _mp_holistic
            from mediapipe.python.solutions import drawing_utils as _mp_drawing
        except ImportError:
            import mediapipe.solutions.holistic as _mp_holistic
            import mediapipe.solutions.drawing_utils as _mp_drawing
    return _mp_holistic, _mp_drawing

def mediapipe_detection(image, model):
    """Perform MediaPipe detection on an image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """Extract keypoints from MediaPipe holistic results.
    Returns:
        np.array: Shape (75, 3) containing pose, left hand, and right hand landmarks.
    """
    # Pose: 33 landmarks
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))

    # Left hand: 21 landmarks
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    else:
        lh = np.zeros((21, 3))

    # Right hand: 21 landmarks
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
    else:
        rh = np.zeros((21, 3))

    return np.vstack([pose, lh, rh])

def draw_styled_landmarks(image, results):
    """Draw styled landmarks on the image."""
    mp_holistic, mp_drawing = _get_mp()
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
