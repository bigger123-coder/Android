from pathlib import Path
import numpy as np

class FaceDatabase:
    def __init__(self, database_path):
        """Initialize an in-memory face database backed by a directory.

        Args:
            database_path: Directory containing one subdirectory per person,
                with face embeddings stored as NumPy files.
        """
        self.database_path = Path(database_path)

        # Database in RAM
        self.database = {}

        # Array containing all vectors in the database
        self.all_embedding = []
        # List of user names in the database
        self.lables = []

    def save(self):
        """Persist the in-memory database to disk.

        This method is currently a placeholder and does not write any data.
        """
        pass

    def load(self):
        """Load all person folders and NumPy embeddings into memory.

        Each embedding is added to both the per-person database and the flat
        lists used by :meth:`search`. Existing in-memory entries are retained.
        """
        # Iterate through each person's folder
        for person_folder in self.database_path.iterdir():

            if not person_folder.is_dir():
                continue

            person_name = person_folder.name
            self.database[person_name] = []

            # Iterate through each person's embeddings
            for embedding_file in person_folder.glob("*.npy"):
                embedding = np.load(embedding_file)

                self.all_embedding.append(embedding)
                self.lables.append(person_name)

                self.database[person_name].append(embedding)

        if len(self.all_embedding) > 0:
            self.all_embedding_ar = np.array(self.all_embedding, dtype=np.float32)
            if self.all_embedding_ar.ndim == 3:
                self.all_embedding_ar = self.all_embedding_ar.squeeze(axis=1)


    def add_person(self):
        """Add a person to the database.

        This method is currently a placeholder and does not modify the database.
        """
        pass

    def get_person(self):
        """Retrieve a person from the database.

        This method is currently a placeholder and returns no value.
        """
        pass

    def search(self, query_embed):
        """Find the closest stored face embedding by cosine similarity.

        Args:
            query_embed: Embedding vector to compare with stored embeddings.

        Returns:
            A dictionary containing the matched name, similarity score, and a
            boolean ``know`` flag indicating whether the threshold was met.
        """
        # --- BƯỚC BẢO VỆ: KIỂM TRA TÍNH HỢP LỆ CỦA DATABASE ---
        if self.all_embedding is None or len(self.all_embedding) == 0:
            return {
                "name": None,
                "score": 0.0,
                "know": False
            }

        # --- BƯỚC BẢO VỆ: KIỂM TRA TÍNH HỢP LỆ CỦA QUERY_EMBED ---
        if query_embed is None:
            return {
                "name": None,
                "score": 0.0,
                "know": False
            }

        query_embed = np.squeeze(np.array(query_embed, dtype=np.float32))

        # Kiểm tra nếu query_embed sai shape
        if query_embed.ndim != 1 or query_embed.shape[0] != 512:
            return {
                "name": None,
                "score": 0.0,
                "know": False
            }

        # Kiểm tra xem có NaN hoặc Inf trong query_embed
        if not np.isfinite(query_embed).all():
            return {
                "name": None,
                "score": 0.0,
                "know": False
            }

        # Tính norm của query và chuẩn hóa query để dùng phép Dot Product
        norm_query = np.linalg.norm(query_embed)
        if norm_query == 0.0:
            return {
                "name": None,
                "score": 0.0,
                "know": False
            }

        threshold = 0.2121

        # Use squeeze to ensure query_embed has shape (512,)
        query_embed = np.squeeze(query_embed)

        # Normalize database_embed to unit vector
        norm_db = np.linalg.norm(self.all_embedding_ar, axis=1, keepdims=True)
        norm_db = np.where(norm_db == 0, 1e-6, norm_db)
        Normalized_db_embed = self.all_embedding_ar / norm_db

        # Normalize query_embed to unit vector
        norm_query = np.linalg.norm(query_embed)
        normalized_query_embed = query_embed / norm_query
 
        # 1/ Calculate the distance to every vector; this is equivalent to cosine similarity
        distance = np.dot(Normalized_db_embed, normalized_query_embed)

        # 2/ Find the nearest neighbor
        best_idx = np.argmax(distance)

        # 3/ Get the nearest neighbor's name and score using its index
        best_person = self.lables[best_idx]
        best_score = distance[best_idx]
        
        if best_score >= threshold:
            return {
                "name": best_person,
                "score": float(best_score),
                "know": True
            }
        else:
            return {
                "name": None,
                "score": float(best_score),
                "know": False
            }