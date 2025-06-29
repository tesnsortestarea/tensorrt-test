import numpy as np
import faiss
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional
import os
import pickle

# Define the embedding dimension
EMBEDDING_DIM = 128


def GenerateSampleData() -> List:

    # Generate and add 10 sample products
    sample_products_data = []
    for i in range(1, 11):
        product_id = f"product_{i:03d}"
        text_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        visual_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        product = ProductEmbedding(
            product_id=product_id,
            text_embedding=text_embedding,
            visual_embedding=visual_embedding,
        )
        db.add_product(product)
        sample_products_data.append(product)

    return sample_products_data


class ProductEmbedding:
    """Class to represent a product with both textual and visual embeddings."""

    def __init__(self, product_id: str, text_embedding: np.ndarray,
                 visual_embedding: Optional[np.ndarray] = None):
        """
            product_id: Unique identifier for the product
            text_embedding: Text embedding vector for the product
            visual_embedding: Visual embedding vector for the product
        """
        self.product_id = product_id
        self.text_embedding = text_embedding
        self.visual_embedding = visual_embedding

    def get_combined_embedding(self, text_weight: float = 0.5) -> np.ndarray:
        """
        Get a combined embedding that merges text and visual embeddings.

        Args:
            text_weight: Weight for text embedding (0.0 to 1.0)
                         Visual weight will be (1 - text_weight)

        Returns:
            Combined embedding vector
        """
        if self.visual_embedding is None:
            return self.text_embedding

        # Normalize embeddings before combining
        norm_text = self.text_embedding / np.linalg.norm(self.text_embedding)
        norm_visual = self.visual_embedding / \
            np.linalg.norm(self.visual_embedding)

        # Combine embeddings with weights
        combined = text_weight * norm_text + (1 - text_weight) * norm_visual

        # Normalize the result
        return combined / np.linalg.norm(combined)


class FAISSVectorDB:
    """Vector database implementation using FAISS."""

    def __init__(self, dimension: int):
        """
        Initialize the FAISS vector database.
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension

        # Initialize FAISS index Flat
        self.index = faiss.IndexFlatL2(dimension)

        # Store product information
        self.products: Dict[int, ProductEmbedding] = {}
        self.id_map: Dict[str, int] = {}  # Maps product_id to FAISS index
        self.next_id = 0

    def add_product(self, product: ProductEmbedding, embedding_type: str = "combined") -> None:
        """
        Add a product to the vector database.

        Args:
            product: ProductEmbedding object to add
            embedding_type: Type of embedding to index ("text", "visual", or "combined")
        """
        if embedding_type == "text":
            vector = product.text_embedding
        elif embedding_type == "visual":
            if product.visual_embedding is None:
                raise ValueError("Product has no visual embedding")
            vector = product.visual_embedding
        else:  # combined
            vector = product.get_combined_embedding()

        # Convert to correct format for FAISS
        vector = np.array([vector]).astype(np.float32)

        # Add to FAISS index
        self.index.add(vector)

        # Store product information
        self.products[self.next_id] = product
        self.id_map[product.product_id] = self.next_id

        # Extend Id by one
        self.next_id += 1

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Search for nearest neighbors in the vector database.

            query_vector: Query embedding vector
            k: Number of nearest neighbors to return

        Returns:
            List of tuples (product_id, distance)
        """
        # Ensure query vector has correct shape and type
        query_vector = np.array([query_vector]).astype(np.float32)

        # Search FAISS index
        distances, indices = self.index.search(query_vector, k)

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.products:  # -1 means no result found
                product = self.products[idx]
                results.append(
                    (product.product_id, distances[0][i]))

        return results

    def save(self, filepath: str) -> None:
        """
        Save the vector database to disk.

                filepath: Path to save the database
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")

        # Save product information
        with open(f"{filepath}.products", "wb") as f:
            pickle.dump({
                "products": self.products,
                "id_map": self.id_map,
                "next_id": self.next_id
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'FAISSVectorDB':
        """
        Load a vector database from disk.

            filepath: Path to load the database from

        Returns:
            Loaded FAISSVectorDB instance
        """
        # Load FAISS index
        index = faiss.read_index(f"{filepath}.index")

        # Load product information
        with open(f"{filepath}.products", "rb") as f:
            data = pickle.load(f)

        # Create instance
        db = cls(index.d)
        db.index = index
        db.products = data["products"]
        db.id_map = data["id_map"]
        db.next_id = data["next_id"]

        return db
