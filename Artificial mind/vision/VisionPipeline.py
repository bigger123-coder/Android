import cv2
import time
import mediapipe as mp
import threading

from vision.Face.Detector import FaceDetector
from vision.Face.Recognition import FaceRecognizer

from vision.Hand.Gesture_recognizer import GestureRecognizer

from vision.Object.Detection import ObjectDetection

from config.vision_config import ModelsConfig

from STT_and_TTS.VAD import VADlistener

from STT_and_TTS.State_machine import STTStateMachine

class VisionPipeline:
    def __init__(self):
        self.models = ModelsConfig()

        # Face
        self.face_detector = FaceDetector(self.models.face_detector_model_path)
        self.face_recognizer = FaceRecognizer(self.models.face_recognizer_model_path)

        # Gesture
        self.gesture = GestureRecognizer(self.models.hand_gesture_recognizer_path)
        self.gesture_recognizer = self.gesture.create_recognizer()

        # Object
        self.detector = ObjectDetection(self.models.object_detection_model_path)
        self.object_detector = self.detector.object_detector()

        # Shutdown protocol
        self._stop_event = threading.Event()
        self._audio_thread = None

         # State machine
        self.state_machine = STTStateMachine(stop_event=self._stop_event)

        self._start_audio_thread()

        self.user_name = None

    # ==========================
    # PROCESSING
    # ==========================
    def _start_audio_thread(self):
        def run_fsm():
            while not self._stop_event.is_set():
                self.state_machine.start_listening()

                self.user_name = self.state_machine.recognized_name

        self._audio_thread = threading.Thread(target=run_fsm, daemon=True)
        self._audio_thread.start()


    def process(self, frame):
        # ==========================
        # Face
        # ==========================
        frame, kpsses, bboxes = self.face_detector.face_detect(frame)  # Detect and draw a bounding box for each face in the camera; kpsses is the list of key points

        if self.user_name is not None:
            # Mode 1: Register a face when user_name is provided
            self.face_recognizer.saving_embedding(frame, kpsses, self.user_name)
            results = []

            # Reset variable user_name to None
            self.user_name = None
        else:
            # Mode 2: Automatically recognize known people when no registration is provided
            frame, results = self.face_recognizer.recognize_faces(frame, kpsses, bboxes)

        known_faces = [bool(result.get("know", False)) for result in results]
        self.state_machine.update_known_faces(known_faces)

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
        # Thực thi shutdown prtocol 
        self._stop_event.set()

        # Nếu VADlistener đang bị block trong InputStream, có thể cần cơ chế ngắt queue/stream ở đây 
        # hoặc đợi luồng audio kết thúc an toàn với timeout ngắn:
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)

        self.gesture_recognizer.close()
        self.object_detector.close()
        self.face_detector.close()
        self.face_recognizer.close()