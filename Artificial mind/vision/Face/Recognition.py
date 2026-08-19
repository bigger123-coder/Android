from insightface.model_zoo import get_model
import cv2
from pathlib import Path
import numpy as np

from vision.Face.Alignment import face_alighment
from vision.Face.Detector import FaceDetector

from config.vision_config import VisionConfig

class FaceRecognizer:
    def __init__(self, model_path, ctx_id=0):
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=ctx_id)

        self.input_size = self.model.input_size
        self.input_mean = self.model.input_mean
        self.input_std = self.model.input_std
        self.session = self.model.session
        self.output_names = self.model.output_names
        self.input_name = self.model.input_name

        # Tạo đường dẫn đển folder embedding
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.EMBED_FOLDER = self.BASE_DIR / "memory" / "face embedding"

        # State
        self.frame_id = 0

    def get_embedding(self, aligned_face):
        """
            Trích xuất embedding từ khuôn mặt đã alignment.

            Parameters
            ----------
            aligned_face : numpy.ndarray hoặc list
                Ảnh khuôn mặt đã alignment, kích thước 112x112.

            Returns
            -------
            numpy.ndarray
                Vector embedding có kích thước (1, 512).
            """
        if not isinstance(aligned_face, list):
            aligned_face = [aligned_face]

        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(aligned_face, 1 / self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        embedding = self.session.run(self.output_names, {self.input_name:blob})[0]

        return embedding.tolist()

    def saving_embedding(self, frame, kpsses, user_name):
        self.frame_id += 1
        
        if len(kpsses) > 0 and self.frame_id % 30 ==0 and user_name is not None: # Lấy embedding khuôn mặt mỗi 30 frame
            # Tự động tạo folder theo tên người dùng
            user_folder = self.EMBED_FOLDER / user_name.strip()
            user_folder.mkdir(parents=True, exist_ok=True)
        
            # Đếm số file embedding có sẵn
            current_embed_file = len(list(user_folder.glob("*.npy")))
        
            for kpss in kpsses:
                # 1/ Căn chỉnh khuôn mặt và trích xuất embedding
                aligned_frame = face_alighment(frame, kpss)
                embedding = self.get_embedding(aligned_frame)
        
                # 2/ Lưu embedding thành file
                if current_embed_file < 10:
                    file_name = user_name.lower().replace(" ", "_")
                    saving_file = f"{file_name}_{current_embed_file}.npy"
                    saving_path = user_folder / saving_file
        
                    np.save(saving_path, embedding)
                    print(f"Đã lưu embedding {user_name} thứ {current_embed_file}")
                    current_embed_file += 1

    def compare_embed(self, embed1, embed2):
        """input: embed_1, embeded_2
        output: điểm số khi so sánh 2 embedding, nếu gần giống nhau thì là 1 người"""

        score = self.model.compute_sim(embed1, embed2)
        return score