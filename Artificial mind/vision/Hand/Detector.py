import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path):
        """Initialize a hand landmark detector with a model path.

        Args:
            model_path: Path to the MediaPipe hand landmarker task file.
        """
        self.model = model_path

        # Store the latest result
        self.hand_landmarks = []

    def result_callback(self, result, output_image, timestamp_ms):
        """Store hand landmarks returned by the asynchronous detector.

        Args:
            result: MediaPipe hand landmark result.
            output_image: Processed image supplied by MediaPipe.
            timestamp_ms: Result timestamp in milliseconds.
        """
        self.hand_landmarks = result.hand_landmarks

    def hand_landmarker(self):
        """Create and configure a live-stream hand landmarker.

        Returns:
            A configured MediaPipe ``HandLandmarker`` instance.
        """
        # Initialize MediaPipe settings
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
        """Draw detected hand landmarks and joint connections on a frame.

        Args:
            frame: BGR image modified in place with hand annotations.
        """
        h, w, _ = frame.shape

        # List of connections
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
                # Draw 21 landmarks
                for landmark in hand:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(frame, (x,y), 5, (0,0,255), -1)
                # Draw connections between joints
                for start, end in HAND_CONNECTIONS:
                    # Coordinates of the start point
                    x1 = int(hand[start].x * w)
                    y1 = int(hand[start].y * h)
                    # Coordinates of the destination point
                    x2 = int(hand[end].x * w)
                    y2 = int(hand[end].y * h)

                    cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)