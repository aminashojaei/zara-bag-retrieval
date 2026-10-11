from dataclasses import dataclass
from typing import Protocol

import numpy as np
from .data import ImageRecord


@dataclass(frozen=True)
class ProductHit:
    product_id: str
    image_path: str
    score: float
    gallery_rank: int


class SearchIndex(Protocol):
    ntotal: int
    d: int
    def search(self, vectors: np.ndarray, k: int): ...


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    a = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("Query embedding contains NaN/Inf")
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        raise ValueError("Query embedding is empty")
    return np.ascontiguousarray((a / norm)[None, :])


def top_products(index: SearchIndex, gallery: list[ImageRecord], query: np.ndarray, k: int = 3) -> list[ProductHit]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if index.ntotal != len(gallery):
        raise ValueError("FAISS index and gallery inventory do not have matching lengths")
    vector = normalize_vector(query)
    if vector.shape[1] != index.d:
        raise ValueError(f"Expected embedding dim {index.d}, got {vector.shape[1]}")
    if not gallery:
        return []
    target = min(k, len({r.product_id for r in gallery}))
    fetch = min(len(gallery), max(k * 8, 32))
    while True:
        scores, indices = index.search(vector, fetch)
        hits = []
        seen = set()
        for image_rank, (i, score) in enumerate(zip(indices[0], scores[0]), 1):
            if i < 0:
                continue
            record = gallery[int(i)]
            if record.product_id in seen:
                continue
            seen.add(record.product_id)
            hits.append(ProductHit(record.product_id, str(record.path), float(score), image_rank))
            if len(hits) == target:
                return hits
        if fetch >= len(gallery):
            return hits
        fetch = min(len(gallery), fetch * 2)
