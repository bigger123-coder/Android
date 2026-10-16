import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ObjectDetection:
    def __init__(self, model_path):
        """Initialize an object detector with a MediaPipe task file.

        Args:
            model_path: Path to the object detector task file.
        """
        self.model = model_path

        self.detections = []

    def result_callback(self, result, output_image, timestamp_ms):
        """Store object detections returned by the asynchronous detector.

        Args:
            result: MediaPipe object detection result.
            output_image: Processed image supplied by MediaPipe.
            timestamp_ms: Result timestamp in milliseconds.
        """
        self.detections = result.detections

    def object_detector(self):
        """Create a live-stream object detector excluding people.

        Returns:
            A configured MediaPipe ``ObjectDetector`` instance.
        """
        base_options = python.BaseOptions(model_asset_path=self.model)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, 
            max_results=1,
            score_threshold=0.5,
            running_mode=vision.RunningMode.LIVE_STREAM,
            category_denylist=["person"],
            result_callback=self.result_callback)

        object_detector = vision.ObjectDetector.create_from_options(options)

        return object_detector

    def draw_object(self, frame):
        """Draw detected object boxes, labels, and confidence scores.

        Args:
            frame: BGR image modified in place with object annotations.
        """
        h, w, _ = frame.shape

        for detection in self.detections:
            bbox = detection.bounding_box

            # Variables containing the x, y, height, and width of a detected object's bounding box
            x = bbox.origin_x
            y = bbox.origin_y
            height = bbox.height
            width = bbox.width

            cv2.rectangle(frame, (x,y), (x + width,y + height), (0,255,0), 2)

            if len(detection.categories) > 0:
                category = detection.categories[0]
                label = category.category_name
                score = category.score

                text = f"{label} {score:.2f}"

                cv2.putText(frame, text, ( int(x) , int(y-10) ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)