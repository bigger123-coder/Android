import cv2
import time
import mediapipe as mp

from vision.Face.Detector import FaceDetector
from vision.Face.Recognition import FaceRecognizer

from vision.Hand.Gesture_recognizer import GestureRecognizer

from vision.Object.Detection import ObjectDetection

from config.vision_config import VisionConfig

class VisionPipeline:
    def __init__(self):
        self.models = VisionConfig()

        # Face
        self.face_detector = FaceDetector(self.models.face_detector_model_path)
        self.face_recognizer = FaceRecognizer(self.models.face_recognizer_model_path)

        # Gesture
        self.gesture = GestureRecognizer(self.models.hand_gesture_recognizer_path)
        self.gesture_recognizer = self.gesture.create_recognizer()

        # Object
        self.detector = ObjectDetection(self.models.object_detection_model_path)
        self.object_detector = self.detector.object_detector()

    # ==========================
    # PROCESSING
    # ==========================
    def process(self, frame, user_name):
        # ==========================
        # Face
        # ==========================
        frame, kpsses, bboxes = self.face_detector.face_detect(frame) # Phát hiện và vẽ bbox cho từng khuôn mặt xuất hiện trong camera, kpsses là danh sách key points

        if user_name is not None:
            # Chế độ 1: Đăng ký khuôn mặt khi có truyền vào user_name
            self.face_recognizer.saving_embedding(frame, kpsses, user_name)
        else:
            # Chế độ 2: Tự động nhận diện (search) tất cả người quen khi không đăng ký
            frame, results = self.face_recognizer.recognize_faces(frame, kpsses, bboxes)

        # ==========================
        # Gesture
        # ==========================
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
        timestamp_ms = int(time.time() * 1000)
            
        self.gesture_recognizer.recognize_async(mp_image, timestamp_ms)

        self.gesture.draw_hand(frame)
        self.gesture.draw_gesture(frame)
            
        # ==========================
        # Object
        # ==========================
            
        self.object_detector.detect_async(mp_image, timestamp_ms)
            
        self.detector.draw_object(frame)

        return frame

    def close(self):
        self.gesture_recognizer.close()
        self.object_detector.close()
        self.face_detector.close()
        self.face_recognizer.close()