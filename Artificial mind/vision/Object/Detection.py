import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ObjectDetection:
    def __init__(self, model_path):
        self.model = model_path

        self.detections = []

    def result_callback(self, result, output_image, timestamp_ms):
        self.detections = result.detections

    def object_detector(self):
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
        """
        Vẽ Bounding Box và tên object lên frame
        """
        h, w, _ = frame.shape

        for detection in self.detections:
            bbox = detection.bounding_box

            # Các biến chứa x, y, height, width của bouding của 1 object khi được phát hiện
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