import cv2
import numpy as np

# Lazy MediaPipe imports
_mp_hands = None
_mp_drawing = None

def _get_mp():
    global _mp_hands, _mp_drawing
    if _mp_hands is None:
        import mediapipe as mp
        try:
            from mediapipe.python.solutions import hands as _mp_hands
            from mediapipe.python.solutions import drawing_utils as _mp_drawing
        except ImportError:
            import mediapipe.solutions.hands as _mp_hands
            import mediapipe.solutions.drawing_utils as _mp_drawing
    return _mp_hands, _mp_drawing

def mediapipe_detection(image, model, last_roi=None):
    """Perform MediaPipe detection with Dynamic ROI."""
    from app.config import MP_INTERNAL_WIDTH, MP_INTERNAL_HEIGHT, MP_USE_CROP, MP_CROP_SIZE
    
    h, w, _ = image.shape
    
    # Defaults
    crop_x, crop_y = 0, 0
    crop_w, crop_h = w, h
    
    process_image = image
    
    if MP_USE_CROP:
        if last_roi:
            # ROI format: (x, y, w, h)
            crop_x, crop_y, crop_w, crop_h = last_roi
            
            # Add margin and ensure bounds
            margin = 50
            crop_x = max(0, crop_x - margin)
            crop_y = max(0, crop_y - margin)
            crop_w = min(w - crop_x, crop_w + 2*margin)
            crop_h = min(h - crop_y, crop_h + 2*margin)
            
            # Defensive check
            if crop_w > 10 and crop_h > 10:
                process_image = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            else:
                # Invalid crop (too small), reset to full frame
                process_image = image
                # Reset ROI for next frame to recover
                last_roi = None
        else:
            # Fallback to center crop if enabled in config but no tracking yet,
            # OR just process full frame. Let's do center crop for initial detection speedup
            # if we are sure the user is centered. Otherwise full frame.
            # Let's stick to full frame for initial detection to be safe.
            pass

    # Resize for speed
    # Note: If we cropped, we should resize carefully.
    # Actually, MediaPipe recommends keeping aspect ratio. 
    # Let's just pass the processed image directly if it's small enough, otherwise downscale.
    if process_image.shape[1] > MP_INTERNAL_WIDTH:
        scale = MP_INTERNAL_WIDTH / process_image.shape[1]
        dim = (MP_INTERNAL_WIDTH, int(process_image.shape[0] * scale))
        input_image = cv2.resize(process_image, dim)
    else:
        input_image = process_image # 3. Process
    msg = f"[DEBUG] MP Shape: {input_image.shape}"
    t0 = time.time()
    
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = model.process(rgb_image)
    
    dt = (time.time() - t0) * 1000
    print(f"{msg} | Time: {dt:.1f}ms")
    
    # Calculate New ROI based on detections
    # With Hands model, we iterate multi_hand_landmarks
    new_roi = None
    
    # Helper to find bounding box of all landmarks
    x_min, y_min = 1.0, 1.0
    x_max, y_max = 0.0, 0.0
    found = False
    
    if results.multi_hand_landmarks:
        found = True
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x_min = min(x_min, min(xs))
            y_min = min(y_min, min(ys))
            x_max = max(x_max, max(xs))
            y_max = max(y_max, max(ys))
    
    if found:
        # Convert relative coords back to absolute pixels relative to the INPUT image (process_image)
        ph, pw, _ = input_image.shape
        
        # Denormalize relative to input_image size
        abs_x = int(x_min * pw)
        abs_y = int(y_min * ph)
        abs_w = int((x_max - x_min) * pw)
        abs_h = int((y_max - y_min) * ph)
        
        # If we resized input_image, scale back to process_image size
        if process_image.shape[1] != input_image.shape[1]:
             scale_factor = process_image.shape[1] / input_image.shape[1]
             abs_x = int(abs_x * scale_factor)
             abs_y = int(abs_y * scale_factor)
             abs_w = int(abs_w * scale_factor)
             abs_h = int(abs_h * scale_factor)
        
        # If we cropped process_image from original, add offsets
        final_x = crop_x + abs_x
        final_y = crop_y + abs_y
        
        new_roi = (final_x, final_y, abs_w, abs_h)
        
    return image, results, new_roi

def extract_keypoints(results):
    """Extract keypoints from MediaPipe Hands results.
    Returns:
        np.array: Shape (75, 3) containing pose (zeroed), left hand, and right hand landmarks.
    """
    # Pose: 33 landmarks (ZERO - Not available in Hands model)
    pose = np.zeros((33, 3))

    # Hands
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, classification in enumerate(results.multi_handedness):
            # label is 'Left' or 'Right'
            # Note: MediaPipe assumes mirrored image by default? 
            # If front camera, 'Left' label usually appears on the right side of screen (user's left)
            # Let's trust the label.
            label = classification.classification[0].label
            landmarks = results.multi_hand_landmarks[idx]
            
            # Extract (21, 3)
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            if label == 'Left':
                lh = lm_array
            else:
                rh = lm_array

    return np.vstack([pose, lh, rh])

def draw_styled_landmarks(image, results):
    """Draw styled landmarks on the image."""
    mp_hands, mp_drawing = _get_mp()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
