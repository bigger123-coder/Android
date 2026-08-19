from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_PATH = BASE_DIR / "AI model" / "Vision model"  # Đường dẫn tới thư mục chứa các model

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

class VisionConfig:
    def __init__(self):
        self.face_detector_model_path = str(FACE_MODELS_PATH / "det_10g.onnx")  # Model phát hiện khuôn mặt
        self.face_recognizer_model_path = str(FACE_MODELS_PATH / "w600k_r50.onnx")  # Model lấy embedding khuôn mặt
        self.hand_detector_model_path = str(HAND_MODELS_PATH / "hand_landmarker.task")  # Model phát hiện bàn tay
        self.hand_gesture_recognizer_path = str(HAND_MODELS_PATH /"gesture_recognizer.task") # Model phát hiện cử chỉ bàn tay
        self.object_detection_model_path = str(OBJECT_MODELS_PATH / "efficientdet_lite2.tflite")  # Model phát hiện vật thể