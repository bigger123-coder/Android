from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_PATH = BASE_DIR / "AI model" / "Vision model"  # Path to the directory containing the models

# ==========================
# FACE MODEL
# ==========================
FACE_MODELS_PATH = MODELS_PATH / "buffalo_l"

# ==========================
# HAND MODEL
# ==========================
HAND_MODELS_PATH = MODELS_PATH / "Hand"

# ==========================
# OBJECT MODEL
# ==========================
OBJECT_MODELS_PATH = MODELS_PATH / "Object"

class ModelsConfig:
    def __init__(self):
        """Build paths to the face, hand, and object detection models."""
        self.face_detector_model_path = str(FACE_MODELS_PATH / "det_10g.onnx")  # Face detection model
        self.face_recognizer_model_path = str(FACE_MODELS_PATH / "w600k_r50.onnx")  # Face embedding model
        self.hand_detector_model_path = str(HAND_MODELS_PATH / "hand_landmarker.task")  # Hand detection model
        self.hand_gesture_recognizer_path = str(HAND_MODELS_PATH /"gesture_recognizer.task") # Hand gesture detection model
        self.object_detection_model_path = str(OBJECT_MODELS_PATH / "efficientdet_lite2.tflite")  # Object detection model