from config import gpu_runtime
from insightface.model_zoo import get_model
import cv2, math
from pathlib import Path
import numpy as np

from vision.Face.Alignment import face_alighment
from vision.Face import Frame_filter
from vision.Face.FaceDatabase import FaceDatabase

class FaceRecognizer:
    def __init__(self, model_path, ctx_id=0):
        """Load and prepare the face recognition model.

        Args:
            model_path: Path to the face recognition model.
            ctx_id: InsightFace execution context identifier.
        """
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=ctx_id)

        self.input_size = self.model.input_size
        self.input_mean = self.model.input_mean
        self.input_std = self.model.input_std
        self.session = self.model.session
        self.output_names = self.model.output_names
        self.input_name = self.model.input_name

        # Create the path to the embedding folder
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.EMBED_FOLDER = self.BASE_DIR / "memory" / "face embedding"

        # State
        self.frame_id = 0

        # initiate modules
        self.face_database = FaceDatabase(self.EMBED_FOLDER)
        self.face_database.load()

    def is_angle_close(self, value, threshold_list, rel_tol):
        for target in threshold_list:
            if math.isclose(value, target, rel_tol=rel_tol):
                return True
        return False
    
    def get_embedding(self, aligned_face):
        """Extract an embedding from an aligned face.

        Args:
            aligned_face: A 112 by 112 aligned face image or a list of such
                images.

        Returns:
            A flattened NumPy embedding vector with 512 values.
        """
        if not isinstance(aligned_face, list):
            aligned_face = [aligned_face]

        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(aligned_face, 1 / self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        embedding = self.session.run(self.output_names, {self.input_name: blob})[0]

        return embedding.flatten()

    def saving_embedding(self, frame, kpsses, user_name):
        """Save a valid face embedding every 30 frames during enrollment.

        Args:
            frame: Current BGR camera frame.
            kpsses: Facial landmark arrays detected in the frame.
            user_name: Person's name used to select the output directory.

        The frame must pass blur and pose checks. At most ten embeddings are
        stored for each user.
        """
        # Orientation angle thresholds (yaw asymmetry and roll)
        yaw_threshold_list = [0.0025, 0.0996, 0.1587, 0.2241, 0.2748, 0.3006]
        roll_threshold_list = [1.82, 5.94, 8.47, 12.89, 17.0, 20.23]

        self.frame_id += 1

        if len(kpsses) == 0 or user_name is None:
            return 

        # Check the sharpness of the frame
        is_sharp = Frame_filter.LaplacianFilter(frame)
        if not is_sharp:
            return  # Exit if the image is blurry

        # Capture an embedding every 30 frames
        if self.frame_id % 30 == 0:
            # Automatically create a folder using the user's name
            user_folder = self.EMBED_FOLDER / user_name.strip()
            user_folder.mkdir(parents=True, exist_ok=True)
            
            # Count the existing embedding files
            current_embed_file = len(list(user_folder.glob("*.npy")))
            if current_embed_file >= 10:
                return  

            # Iterate over each detected face
            for kpss in kpsses:
                ready_to_capture, yaw_asym, roll = Frame_filter.PoseFiltering(kpss)

                # Compare yaw_asym and roll with approximately values
                is_yaw_matched = self.is_angle_close(yaw_asym, yaw_threshold_list, rel_tol=0.05)
                is_roll_matched = self.is_angle_close(abs(roll), roll_threshold_list, rel_tol=0.05)

                # Condition: the angle is within the safe range and passes the filter
                if ready_to_capture and (is_yaw_matched or is_roll_matched):
                    # Align the face and extract its embedding
                    aligned_frame = face_alighment(frame, kpss)
                    embedding = self.get_embedding(aligned_frame)
                
                    # Save the embedding as an .npy file
                    file_name = user_name.lower().replace(" ", "_")
                    saving_file = f"{file_name}_{current_embed_file}.npy"
                    saving_path = user_folder / saving_file
            
                    np.save(saving_path, embedding)
                    print(f"Đã lưu embedding {user_name} thứ {current_embed_file} (Yaw: {yaw_asym:.4f}, Roll: {roll:.2f})")
                    current_embed_file += 1
                    
                    if current_embed_file >= 10:
                        break

    def recognize_faces(self, frame, kpsses, bboxes):
        """Nhận diện tất cả khuôn mặt, vẽ bounding box và hiển thị tên lên frame.

        Args:
            frame: Current BGR camera frame.
            kpsses: Facial landmark arrays detected in the frame.
            bboxes: Bounding box coordinates for each detected face.

        Returns:
            frame: Khung hình đã được vẽ bounding box và tên.
            results: Danh sách kết quả nhận diện.
        """
        if len(kpsses) == 0 or len(bboxes) == 0:
            return frame, []

        recognition_results = []

        # Duyệt qua từng khuôn mặt tương ứng với bboxes và kpsses
        for i, kpss in enumerate(kpsses):
            bbox = bboxes[i]  # Lấy tọa độ bbox của khuôn mặt hiện tại (ví dụ: [x1, y1, x2, y2])
            x1, y1, x2, y2 = map(int, bbox[:4])

            # 1. Căn chỉnh khuôn mặt và lấy vector embedding[cite: 3]
            aligned_frame = face_alighment(frame, kpss)
            embedding = self.get_embedding(aligned_frame)

            # 2. Gọi hàm search từ FaceDatabase để tìm danh tính[cite: 4]
            match_result = self.face_database.search(embedding)

            name = match_result["name"]
            score = match_result["score"]
            know = match_result["know"]

            recognition_results.append({
                "kpss": kpss,
                "bbox": bbox,
                "name": name,
                "score": score,
                "know": know
            })

            # 3. Xác định màu sắc và nội dung chữ dựa vào trạng thái nhận diện
            if know:
                text = f"{name} ({score:.2f})"
                color = (0, 255, 0)  # Xanh lá cho người quen
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Đỏ cho người lạ

            # 4. Vẽ Bounding Box quanh khuôn mặt
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 5. Vẽ text tên ngay phía trên góc trái của Bounding Box
            # Đảm bảo vị trí y không bị tràn lên trên cùng của khung hình
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            
            cv2.putText(
                img=frame,
                text=text,
                org=(x1, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

        return frame, recognition_results

    def close(self):
        """Release references held by the face recognition model."""
        self.session = None
        self.model = None