import cv2
import numpy as np
from skimage.transform import SimilarityTransform

def face_alighment(frame, kpss):
    """Align a face to the recognizer's standard 112 by 112 image layout.

    Args:
        frame: Source image containing the face.
        kpss: Five facial landmarks for the eyes, nose, and mouth corners.

    Returns:
        A similarity-transformed face image with dimensions 112 by 112.
    """
    # Standard landmarks for alignment
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    src = kpss.astype(np.float32)

    tform = SimilarityTransform.from_estimate(src, dst)

    M = tform.params[:2]

    aligned = cv2.warpAffine(frame, M, (112,112))

    return aligned