from insightface.model_zoo import get_model
import cv2

class FaceDetector():
    def __init__(self, model_path, ctx_id=0):
        """Load and prepare the InsightFace detection model.

        Args:
            model_path: Path to the face detection model.
            ctx_id: InsightFace execution context identifier.
        """
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=ctx_id, input_size=(640,640))

    def face_detect(self, frame):
        """Detect faces, draw their bounding boxes, and return their landmarks.

        Args:
            frame: BGR image to process. The image is modified in place with
                a rectangle around each detected face.

        Returns:
            A tuple containing the annotated frame and the detected facial
            landmark arrays.
        """
        
        bboxes, kpsses = self.model.detect(frame)

        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4].astype(int)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        return frame, kpsses, bboxes

    def close(self):
        """Release references held by the face detection model."""
        self.model = None