from ...config import gpu_runtime
from insightface.model_zoo import get_model
from pathlib import Path
from .Alignment import face_alighment
from . import Frame_filter
from .FaceDatabase import FaceDatabase
from ...STT_and_TTS.Database_manager import get_next_id, load_person_database, update_person_embed_dir

import numpy as np
import cv2, math

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

        # Create EMBED_FOLDER if it was not created
        self.EMBED_FOLDER.mkdir(parents=True, exist_ok=True)

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
            person_id: Person's id used to select the output directory.

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
            # Load database
            person_database = load_person_database()

            # Tìm xem user_name đã tồn tại trong JSON chưa để dùng lại ID cũ, nếu chưa thì cấp ID mới
            person_id = None
            for pid, pdata in person_database.items():
                saved_name = pdata.get("name")

                if saved_name and saved_name.lower() == user_name.lower().strip():
                    person_id = pid
                    break

            if not person_id:
                person_id = get_next_id(person_database)

            # Automatically create a folder using the user's id
            user_folder = self.EMBED_FOLDER / person_id
            user_folder.mkdir(parents=True, exist_ok=True)

            # Cập nhật thư mục embedding vào database cho person_id này
            update_person_embed_dir(person_id, str(user_folder))
            
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
                    file_name = person_id
                    saving_file = f"{file_name}_{current_embed_file}.npy"
                    saving_path = user_folder / saving_file
            
                    np.save(saving_path, embedding)
                    print(f"Saved embedding for {user_name} #{current_embed_file} (Yaw: {yaw_asym:.4f}, Roll: {roll:.2f})")
                    current_embed_file += 1
                    
                    if current_embed_file >= 10:
                        break

    def recognize_faces(self, frame, kpsses, bboxes):
        """Recognize all faces, draw bounding boxes, and display names on the frame.

        Args:
            frame: Current BGR camera frame.
            kpsses: Facial landmark arrays detected in the frame.
            bboxes: Bounding box coordinates for each detected face.

        Returns:
            frame: Frame with bounding boxes and names drawn on it.
            results: List of recognition results.
        """
        if len(kpsses) == 0 or len(bboxes) == 0:
            return frame, []

        recognition_results = []

        # Iterate through each face corresponding to the bboxes and kpsses
        for i, kpss in enumerate(kpsses):
            bbox = bboxes[i]  # Get the current face bounding box coordinates (for example: [x1, y1, x2, y2])
            x1, y1, x2, y2 = map(int, bbox[:4])

            # 1. Align the face and extract its embedding vector
            aligned_frame = face_alighment(frame, kpss)
            embedding = self.get_embedding(aligned_frame)

            # 2. Use the FaceDatabase search function to find the identity
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

            # 3. Determine the text color and label based on recognition status
            if know:
                text = f"{name} ({score:.2f})"
                color = (0, 255, 0)  # Green for a known person
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Red for an unknown person

            # 4. Draw a bounding box around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 5. Draw the name text just above the upper-left corner of the bounding box
            # Ensure the y position does not overflow beyond the top of the frame
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