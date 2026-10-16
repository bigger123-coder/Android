import cv2
import time
import mediapipe as mp
import threading
import winsound

from pathlib import Path

from .Face.Detector import FaceDetector
from .Face.Recognition import FaceRecognizer
from .Hand.Gesture_recognizer import GestureRecognizer
from .Object.Detection import ObjectDetection

from ..config.vision_config import ModelsConfig

from ..STT_and_TTS.VAD import VADlistener
from ..STT_and_TTS.State_machine import STTStateMachine

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

        # Voice audio path
        self.base_dir = Path(__file__).resolve().parent.parent
        self.voice_audio = self.base_dir / "Voice" / "Greeting_voice.wav"

        # Saving name status 
        self.is_enrolling = False
        self.pending_name = None

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

        # Luôn chạy nhận diện/tìm kiếm trước để biết khung hình chứa ai (Known hay Unknown)
        frame, results = self.face_recognizer.recognize_faces(frame, kpsses, bboxes)

        # Checking is there any unknown face
        has_unknown_face = None

        for face in results:
            if not face.get("know", False):
                has_unknown_face = True
                break

        if has_unknown_face:
            if not self.is_enrolling:
                self.is_enrolling = True
                self.pending_name = None 
                # Mode 1: Register a face when camera identifies an unknown face
                #winsound.PlaySound(self.voice_audio, flags=winsound.SND_ASYNC)
                print("What is your name?")

            # Nếu luồng STT đã bắt được tên mới, lưu vào pending_name để dùng suốt quá trình
            if self.user_name is not None:
                self.pending_name = self.user_name
                self.user_name = None

            # Tiến hành lưu embedding nếu đã có tên người dùng
            if self.pending_name is not None:
                self.face_recognizer.saving_embedding(frame, kpsses, self.pending_name)

        else:
            if self.is_enrolling:
                self.is_enrolling = False
                self.pending_name = None

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