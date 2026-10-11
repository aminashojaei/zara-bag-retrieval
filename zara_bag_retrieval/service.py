from pathlib import Path

from PIL import Image

from .artifacts import ArtifactStore
from .encoders import EncoderPool
from .retrieval import ProductHit, top_products


class VisualSearchService:
    def __init__(self, dataset: Path, results: Path, device: str = "auto"):
        self.store = ArtifactStore(dataset, results)
        self.encoders = EncoderPool(device)
        self.available = self.store.available_models()
        if not self.available:
            raise FileNotFoundError(
                f"No model artifacts in {results}. Expect subfolders dinov2, dinov3, siglip2 "
                "with gallery.faiss, gallery_embeddings.npy, and gallery_inventory.csv. "
                "Run python build_artifacts.py first."
            )
        self.query_by_key = {r.relative_key: r for r in self.store.query_records}

    def search_saved_query(self, model: str, key: str, k: int = 3) -> list[ProductHit]:
        if key not in self.query_by_key:
            raise KeyError(f"Unknown saved query: {key}")
        artifacts = self.store.load(model)
        embedding = self.store.cached_query(model, key)
        if embedding is None:
            with Image.open(self.query_by_key[key].path) as image:
                embedding = self.encoders.image_embedding(model, image)
        return top_products(artifacts.index, artifacts.gallery, embedding, k=k)

    def search_uploaded_image(self, model: str, image: Image.Image, k: int = 3) -> list[ProductHit]:
        artifacts = self.store.load(model)
        embedding = self.encoders.image_embedding(model, image)
        return top_products(artifacts.index, artifacts.gallery, embedding, k=k)

    def get_query_image_path(self, key: str) -> str:
        if key not in self.query_by_key:
            raise KeyError(key)
        return str(self.query_by_key[key].path)
