from insightface.model_zoo import get_model
import cv2

class FaceDetector():
    def __init__(self, model_path, ctx_id=0):
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=ctx_id, input_size=(640,640))

    def face_detect(self, frame):
        """Input: frame từ camera
        frame được hàm face_detector extract bboxes, kpss
        tọa độ của từng bboxes được dùng để vẽ bouding lên frame
        return frame và kpss
                
        CHÚ Ý: 
        - bboxes là danh sách các bouding box được tìm từ frame
        - kpss là danh sách landmark được dùng cho alignment để căn chỉnh khuôn mặt
        """
        
        bboxes, kpss = self.model.detect(frame)

        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4].astype(int)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        return frame, kpss