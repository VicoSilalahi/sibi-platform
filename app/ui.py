import cv2
import numpy as np

def render_ui(image, label, confidence, is_active=False, energy=0.0):
    """Render the inference UI on top of the image."""
    height, width, _ = image.shape
    
    # Define colors
    bg_color = (45, 45, 45)
    text_color = (255, 255, 255)
    rec_color = (0, 0, 255) if is_active else (200, 200, 200)
    bar_color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)

    # Draw top bar background
    cv2.rectangle(image, (0, 0), (width, 80), bg_color, -1)
    
    # Draw label
    cv2.putText(image, f"SIGN: {label.upper()}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2, cv2.LINE_AA)
    
    # Draw confidence bar
    bar_width = int(width * confidence * 0.4)
    cv2.rectangle(image, (width - bar_width - 20, 30), (width - 20, 50), bar_color, -1)
    cv2.putText(image, f"{int(confidence*100)}%", (width - 100, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    # Draw Gating Status (Debug info)
    status_label = "RECORDING" if is_active else "IDLE"
    cv2.circle(image, (30, 100), 8, rec_color, -1)
    cv2.putText(image, f"{status_label} (Energy: {energy:.4f})", (50, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 1, cv2.LINE_AA)

    return image

def render_no_sign(image, is_active=False, energy=0.0):
    """Render 'No sign' state."""
    return render_ui(image, "No sign", 0.0, is_active, energy)
