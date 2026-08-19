import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path):
        self.model = model_path

        # Lưu kết quả mới nhất
        self.hand_landmarks = []

    def result_callback(self, result, output_image, timestamp_ms):
        self.hand_landmarks = result.hand_landmarks

    def hand_landmarker(self):
        # Khởi tạo cài đặt của mediapipe
        base_options = python.BaseOptions(model_asset_path=self.model)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands = 2,
            running_mode = vision.RunningMode.LIVE_STREAM,
            result_callback=self.result_callback
        )

        hand_detector = vision.HandLandmarker.create_from_options(options)

        return hand_detector

    def draw_hand(self, frame):
        h, w, _ = frame.shape

        # Danh sách các đường nối
        HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]

        if self.hand_landmarks:
            for hand in self.hand_landmarks:
                # Vẽ 21 landmark
                for landmark in hand:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(frame, (x,y), 5, (0,0,255), -1)
                # Vẽ đường nối các khớp
                for start, end in HAND_CONNECTIONS:
                    # Cặp tọa độ của điểm bắt đầu(start) 
                    x1 = int(hand[start].x * w)
                    y1 = int(hand[start].y * h)
                    # Cặp tọa độ của điểm đến(end)
                    x2 = int(hand[end].x * w)
                    y2 = int(hand[end].y * h)

                    cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)