import numpy as np
import pytest
from pathlib import Path
from zara_bag_retrieval.data import ImageRecord
from zara_bag_retrieval.retrieval import top_products, normalize_vector


class FakeFlatIP:
    def __init__(self, embeddings):
        self.vectors = np.asarray(embeddings, dtype=np.float32)
        self.ntotal, self.d = self.vectors.shape

    def search(self, vectors, k):
        similarities = np.asarray(vectors) @ self.vectors.T
        indices = np.argsort(-similarities, axis=1)[:, :k]
        scores = np.take_along_axis(similarities, indices, axis=1)
        return scores.astype(np.float32), indices.astype(np.int64)


def make_gallery(products):
    return [ImageRecord(prod, Path(f"/dummy/{prod}_{i}.jpg"), f"{prod}/{i}.jpg")
            for i, prod in enumerate(products)]


def test_three_distinct_products_even_when_one_has_many_images():
    products = ["A"] * 40 + ["B", "C", "D", "E", "F", "G"]
    query = normalize_vector(np.array([1.0, 0.0]))[0]
    embeddings = []
    for i, product in enumerate(products):
        score = (1.0 - i * 0.001) if product == 'A' else (0.93 - (i - 40) * 0.02)
        embeddings.append([score, np.sqrt(max(0., 1 - score**2))])
    hits = top_products(FakeFlatIP(embeddings), make_gallery(products), query, k=3)
    assert len(hits) == 3
    assert [hit.product_id for hit in hits] == ['A', 'B', 'C']
    assert len({h.product_id for h in hits}) == 3
    assert hits[0].score >= hits[-1].score


def test_best_image_represents_product_not_first_image():
    gallery = make_gallery(['A', 'B', 'A', 'C', 'D', 'E'])
    vecs = np.array([[0.1, .995], [.95, .312], [.99, .141], [.9, .436], [.8, .6], [.7, .714]], np.float32)
    hits = top_products(FakeFlatIP(vecs), gallery, np.array([1, 0]), k=3)
    assert hits[0].product_id == 'A'
    assert hits[0].image_path.endswith('A_2.jpg')


def test_returns_fewer_when_catalog_small():
    gallery = make_gallery(['A', 'A', 'B'])
    hits = top_products(FakeFlatIP([[1., 0], [.9, .1], [.5, .5]]), gallery, [1, 0], k=3)
    assert len(hits) == 2


def test_rejects_mismatched_index():
    with pytest.raises(ValueError, match='matching lengths'):
        top_products(FakeFlatIP([[1., 0]]), [], [1, 0])


def test_normalizes_vectors():
    norm = normalize_vector([3., 4.])
    assert norm.shape == (1, 2)
    np.testing.assert_allclose(norm[0], [.6, .8], atol=1e-6)
    with pytest.raises(ValueError):
        normalize_vector([0., 0.])
