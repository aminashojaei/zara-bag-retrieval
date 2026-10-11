from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .config import MODEL_IDS, REQUIRED_GALLERY_FILES
from .data import ImageRecord, read_inventory, scan_dataset_role, assert_inventory_matches


@dataclass
class ModelArtifacts:
    index: object
    gallery: list[ImageRecord]
    query: list[ImageRecord]
    query_embeddings: np.ndarray | None
    embedding_dim: int


class ArtifactStore:
    def __init__(self, dataset_root: Path, results_root: Path):
        self.dataset_root = Path(dataset_root)
        self.results_root = Path(results_root)
        self._cache: dict[str, ModelArtifacts] = {}
        self.query_records = scan_dataset_role(self.dataset_root, "query")
        self.gallery_records = scan_dataset_role(self.dataset_root, "gallery")
        if not self.query_records or not self.gallery_records:
            raise ValueError("At least one query and gallery image is required")

    def available_models(self) -> list[str]:
        return [key for key in MODEL_IDS if all(
            (self.results_root / key / filename).is_file() for filename in REQUIRED_GALLERY_FILES
        )]

    def load(self, key: str) -> ModelArtifacts:
        if key in self._cache:
            return self._cache[key]
        if key not in MODEL_IDS:
            raise ValueError(f"Unknown model: {key}")
        folder = self.results_root / key
        import faiss
        index = faiss.read_index(str(folder / "gallery.faiss"))
        gallery = read_inventory(folder / "gallery_inventory.csv", self.dataset_root, "gallery")
        assert_inventory_matches(self.gallery_records, gallery, f"{key}/gallery_inventory.csv")
        matrix = np.load(folder / "gallery_embeddings.npy", mmap_mode="r")
        if matrix.shape != (index.ntotal, index.d) or len(gallery) != index.ntotal:
            raise ValueError(f"{key}: FAISS index, embedding array, and gallery inventory disagree")
        if index.ntotal:
            for row in {0, index.ntotal // 2, index.ntotal - 1}:
                original = index.reconstruct(int(row))
                if not np.allclose(original, matrix[row], rtol=1e-4, atol=1e-5):
                    raise ValueError(f"{key}: gallery.faiss does not match gallery_embeddings.npy")
        query_file = folder / "query_inventory.csv"
        query_emb_file = folder / "query_embeddings.npy"
        if query_file.exists() and query_emb_file.exists():
            queries = read_inventory(query_file, self.dataset_root, "query")
            assert_inventory_matches(self.query_records, queries, f"{key}/query_inventory.csv")
            qmat = np.load(query_emb_file, mmap_mode="r")
            if qmat.shape != (len(queries), index.d):
                raise ValueError(f"{key}: query inventory and embedding array disagree")
        else:
            queries, qmat = [], None
        obj = ModelArtifacts(index, gallery, queries, qmat, int(index.d))
        self._cache[key] = obj
        return obj

    def cached_query(self, key: str, relative_key: str) -> np.ndarray | None:
        model = self.load(key)
        if model.query_embeddings is None:
            return None
        for i, record in enumerate(model.query):
            if record.relative_key == relative_key:
                return np.asarray(model.query_embeddings[i], dtype=np.float32)
        return None
