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
        threshold = 0.2121

        # Use squeeze to ensure query_embed has shape (512,)
        query_embed = np.squeeze(query_embed)

        # Explicitly convert to a 2D ndarray (N, 512) to avoid shape errors when
        # all_embedding is a list of arrays with shape (1, 512) instead of (512,)
        all_embedding = np.array(self.all_embedding)
        if all_embedding.ndim == 3:
            all_embedding = all_embedding.squeeze(axis=1)

        # Calculate the dot product
        vector_dot = np.dot(self.all_embedding, query_embed)
        # Calculate the lengths of the two vectors
        norm_database = np.linalg.norm(self.all_embedding, axis=1)
        norm_query = np.linalg.norm(query_embed)
 
        # 1/ Calculate the distance to every vector; this is equivalent to cosine similarity
        distance = vector_dot / (norm_database * norm_query)

        # 2/ Find the nearest neighbor
        best_idx = np.argmax(distance)

        # 3/ Get the nearest neighbor's name and score using its index
        best_person = self.lables[best_idx]
        best_score = distance[best_idx]
        
        if best_score >= threshold:
            return {
                "name": best_person,
                "score": best_score,
                "know": True
            }
        else:
            return {
                "name": None,
                "score": best_score,
                "know": False
            }