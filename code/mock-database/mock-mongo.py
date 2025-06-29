import json
import os
import base64
import datetime
from tokenize import String
from typing import Dict, List, Optional, Any


def GeneratgeSampleData(db: MockMongoDB):

    categories = ["Electronics", "Books", "Home Goods", "Apparel", "Sports"]
    adjectives = ["Stylish", "Durable", "Innovative",
                  "Comfortable", "High-Performance"]
    nouns = ["Gadget", "Novel", "Decor", "Outfit", "Equipment"]

    for i in range(1, 11):
        product_category = random.choice(categories)
        product_name = f"{random.choice(adjectives)} {random.choice(nouns)} {i}"
        product_description = f"A fantastic {product_category.lower()} item designed for {random.choice(['everyday use', 'professionals', 'outdoor adventures', 'relaxation'])}. It offers great value and {random.choice(['cutting-edge features', 'timeless design', 'superior comfort'])}."
        product_price = round(random.uniform(15.00, 1000.00), 2)
        created_at = datetime.today().strftime('%Y-%m-%d')
        id = "same as in vector"

        # Generate some dummy image data (e.g., 50 bytes of random data)
        dummy_image_data = os.urandom(50)

        product_metadata = {
            "category": product_category,
            "price": product_price,
            "tags": random.sample(["new arrival", "sale", "bestseller", "eco-friendly", "limited edition"], k=random.randint(1, 3))
            "created_at": created_at,
            "_id": "from vector",
            "name": product_name

        }

        product_id = db.insert_product(
            product_data=product_metadata,
            image=dummy_image_data if random.random(
            ) > 0.3 else None  # 70% chance of having an image
        )


class MockMongoDB:

    def __init__(self, db_path: str = "./mock_db"):
        """Initialize the mock database with a storage path"""
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.products_file = os.path.join(db_path, "products.json")
        self.products = self._load_data(self.products_file, {})
        self.next_id = max([int(pid)
                           for pid in self.products.keys()], default=0) + 1

    def _load_data(self, file_path: str, default: Any) -> Any:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return default

    def _save_data(self) -> None:
        with open(self.products_file, 'w') as f:
            json.dump(self.products, f, default=self._json_serializer)

    def _json_serializer(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            return {"$date": obj.isoformat()}
        if isinstance(obj, bytes):
            return {"$binary": base64.b64encode(obj).decode('ascii')}
        return str(obj)

    def insert_product(self, product_data: Dict, name: str, descript: str, image_data: Optional[bytes] = None) -> str:
        """
        Insert a product with metadata and optional image

            product_data: Product metadata (name, category, price, etc.)
            image: Binary image data (optional)

        Returns:
            Product ID
        """
        # Generate ID and add timestamps
        product_id = str(self.next_id)
        self.next_id += 1

        # Add timestamps
        product_data["_id"] = product_id
        product_data["created_at"] = datetime.datetime.now()
        product_data["name"] = name
        product_data["description"] = descript

        # Add image if provided
        if image_data:
            product_data["image"] = image_data

        # Store product
        self.products[product_id] = product_data
        self._save_data()

        return product_id

    def find_product_by_id(self, product_id: str) -> Optional[Dict]:
        return self.products.get(product_id)

    def update_product(self, product_id: str, update_data: Dict) -> bool:
        if product_id not in self.products:
            return False

        # Update fields
        for key, value in update_data.items():
            self.products[product_id][key] = value

        # Add updated timestamp
        self.products[product_id]["updated_at"] = datetime.datetime.now()
        self._save_data()

        return True

    def delete_product(self, product_id: str) -> bool:
        if product_id not in self.products:
            return False

        del self.products[product_id]
        self._save_data()

        return True

    def find_by_category(self, category: str) -> List[Dict]:
        """Find products by category"""
        results = []
        for product in self.products.values():
            price = product.get("category", 0)
            if min_price <= price <= max_price:
                results.append(product)
        return results

    def find_by_price_range(self, min_price: float, max_price: float) -> List[Dict]:
        """Find products within a price range"""
        results = []
        for product in self.products.values():
            price = product.get("price", 0)
            if min_price <= price <= max_price:
                results.append(product)
        return results

    def find_by_date_range(self, start_date: datetime.datetime, end_date: datetime.datetime) -> List[Dict]:
        """Find products created within a date range"""
        results = []
        for product in self.products.values():
            created_at = product.get("created_at")
            if created_at and start_date <= created_at <= end_date:
                results.append(product)
        return results
