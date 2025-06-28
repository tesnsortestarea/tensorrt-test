import numpy as np
import faiss
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional
import os
import pickle


class ProductEmbedding:
    """Class to represent a product with both textual and visual embeddings."""

    def __init__(self, product_id: str, text_embedding: np.ndarray,
                 visual_embedding: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None):
        """
            product_id: Unique identifier for the product
            text_embedding: Text embedding vector for the product
            visual_embedding: Visual embedding vector for the product
            metadata: Additional product information
        """
        self.product_id = product_id
        self.text_embedding = text_embedding
        self.visual_embedding = visual_embedding
        self.metadata = metadata or {}

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
            List of tuples (product_id, distance, metadata)
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
                    (product.product_id, distances[0][i], product.metadata))

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


# Example usage
def mock_text_embedding(text: str, dim: int = 512) -> np.ndarray:
    """Generate a mock text embedding for demonstration purposes."""
    # In a real application, you would use a text embedding model here
    np.random.seed(hash(text) % 10000)
    return np.random.random(dim).astype(np.float32)


def mock_image_embedding(image_path: str = None, dim: int = 512) -> np.ndarray:
    """Generate a mock image embedding for demonstration purposes."""
    # In a real application, you would use an image embedding model here
    if image_path:
        np.random.seed(hash(image_path) % 10000)
    else:
        np.random.seed(42)
    return np.random.random(dim).astype(np.float32)


def demo():
    """Demonstrate the FAISS vector database with product embeddings."""
    # Initialize vector database
    vector_db = FAISSVectorDB(dimension=512)

    # Create some mock product embeddings
    products = [
        ProductEmbedding(
            product_id="prod001",
            text_embedding=mock_text_embedding(
                "Blue cotton t-shirt with logo"),
            visual_embedding=mock_image_embedding("tshirt_blue.jpg"),
            metadata={"name": "Blue T-Shirt",
                      "category": "Clothing", "price": 19.99}
        ),
        ProductEmbedding(
            product_id="prod002",
            text_embedding=mock_text_embedding("Red leather wallet"),
            visual_embedding=mock_image_embedding("wallet_red.jpg"),
            metadata={"name": "Leather Wallet",
                      "category": "Accessories", "price": 39.99}
        ),
        ProductEmbedding(
            product_id="prod003",
            text_embedding=mock_text_embedding(
                "Black running shoes with white sole"),
            visual_embedding=mock_image_embedding("shoes_running.jpg"),
            metadata={"name": "Running Shoes",
                      "category": "Footwear", "price": 89.99}
        ),
        ProductEmbedding(
            product_id="prod004",
            text_embedding=mock_text_embedding(
                "Denim jeans with distressed look"),
            visual_embedding=mock_image_embedding("jeans_denim.jpg"),
            metadata={"name": "Distressed Jeans",
                      "category": "Clothing", "price": 59.99}
        ),
        ProductEmbedding(
            product_id="prod005",
            text_embedding=mock_text_embedding("Stainless steel water bottle"),
            visual_embedding=mock_image_embedding("bottle_steel.jpg"),
            metadata={"name": "Water Bottle",
                      "category": "Accessories", "price": 24.99}
        ),
    ]

    # Add products to the database
    for product in products:
        vector_db.add_product(product)

    # Perform a search
    query = mock_text_embedding("blue shirt with logo")
    results = vector_db.search(query, k=3)

    print("Search results for 'blue shirt with logo':")
    for product_id, distance, metadata in results:
        print(f"Product: {metadata['name']} (ID: {product_id})")
        print(f"Distance: {distance:.4f}")
        print(f"Price: ${metadata['price']}")
        print(f"Category: {metadata['category']}")
        print("-" * 40)

    # Save and load the database
    vector_db.save("./product_vector_db")
    loaded_db = FAISSVectorDB.load("./product_vector_db")

    # Verify loaded database
    results_loaded = loaded_db.search(query, k=3)
    print("\nVerifying loaded database results match:")
    match = all(r1[0] == r2[0] and abs(r1[1] - r2[1]) <
                1e-5 for r1, r2 in zip(results, results_loaded))
    print(f"Results match: {match}")


if __name__ == "__main__":
    demo()
