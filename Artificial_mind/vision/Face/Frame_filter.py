import cv2
import numpy as np

def LaplacianFilter(frame):
    """Determine whether an image is sharp enough for face processing.

    Args:
        frame: BGR image to evaluate.

    Returns:
        ``True`` when the Laplacian variance reaches the sharpness threshold;
        otherwise, ``False``.
    """
    # ======================
    # LAPLACIAN
    # ======================
    laplacian_threshold = 50.0
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 1/ Convert the frame to grayscale
    gray_frame_matrix = cv2.Laplacian(gray_frame, cv2.CV_64F) # 2/ Calculate the Laplacian matrix
    blur_score = gray_frame_matrix.var() # 3/ Calculate the variance to determine whether the frame is blurry or sharp
    
    if blur_score >= laplacian_threshold:
        return True
    else:
        return False

def PoseFiltering(kpss):
    """Check whether facial landmarks represent an acceptable head pose.

    Args:
        kpss: Five landmarks for the eyes, nose, and mouth corners.

    Returns:
        A tuple containing the pose decision, yaw asymmetry, and roll angle
        threshold used by the filter.
    """
    left_eye = np.array(kpss[0])
    right_eye = np.array(kpss[1])
    nose_tip = np.array(kpss[2])
    left_mouth_corner = np.array(kpss[3])
    right_mouth_corner = np.array(kpss[4])

    # Maximum thresholds when the user tilts or turns their head
    yaw_max = 0.373 
    roll_max = 41.81

    # Calculate the distances from the nose to both eyes to detect left or right head rotation
    distance_right = np.linalg.norm(nose_tip - right_eye)
    distance_left = np.linalg.norm(nose_tip - left_eye)
    yaw_asym = abs(distance_right - distance_left) / max(distance_right, distance_left, 1e-6)

    # Calculate the angle between the eyes to detect left or right head tilt
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    roll = np.degrees(np.arctan2(dy, dx))

    return yaw_asym <= yaw_max or abs(roll) <= roll_max, yaw_asym, roll