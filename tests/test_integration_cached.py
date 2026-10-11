import csv
import sys
import types

import numpy as np
from PIL import Image

from zara_bag_retrieval.service import VisualSearchService


class FlatIPStub:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, np.float32)
        self.ntotal, self.d = self.matrix.shape

    def reconstruct(self, i):
        return self.matrix[i].copy()

    def search(self, vectors, k):
        scores = np.asarray(vectors) @ self.matrix.T
        indices = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, indices, axis=1), indices


def test_cached_search_uses_stored_embeds_and_returns_distinct_products(tmp_path, monkeypatch):
    data = tmp_path / 'zara_retrieval_dataset'
    result = tmp_path / 'artifacts' / 'siglip2'
    result.mkdir(parents=True)
    product_names = [f'img {i:03d}' for i in range(1, 7)]
    gallery_rows = []
    for i, product in enumerate(product_names):
        p = data / 'gallery' / product / 'photo.png'
        p.parent.mkdir(parents=True)
        Image.new('RGB', (10, 10), (i*25, 0, 0)).save(p)
        gallery_rows.append((product, p))
    query_path = data / 'query' / product_names[0] / 'query.png'
    query_path.parent.mkdir(parents=True)
    Image.new('RGB', (10, 10), 'red').save(query_path)
    vec = np.asarray([
        [1.0, 0.0], [.97, .2431], [.9, .4359], [.8, .6], [.7, .7141], [.6, .8]
    ], np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    np.save(result / 'gallery_embeddings.npy', vec)
    np.save(result / 'query_embeddings.npy', np.asarray([[1.0, 0.0]], np.float32))
    (result / 'gallery.faiss').write_bytes(b'stub')
    for filename, rows in (
        ('gallery_inventory.csv', gallery_rows),
        ('query_inventory.csv', [(product_names[0], query_path)]),
    ):
        with (result / filename).open('w', newline='', encoding='utf8') as fp:
            w = csv.DictWriter(fp, fieldnames=['product_id','path'])
            w.writeheader()
            w.writerows({'product_id': product_id, 'path': str(path)} for product_id, path in rows)
    fake_faiss = types.SimpleNamespace(read_index=lambda path: FlatIPStub(vec))
    monkeypatch.setitem(sys.modules, 'faiss', fake_faiss)
    service = VisualSearchService(data, tmp_path / 'artifacts')
    hits = service.search_saved_query('siglip2', 'img 001/query.png')
    assert len(hits) == 3
    assert len({h.product_id for h in hits}) == 3
    assert hits[0].product_id == 'img 001'
    assert hits[0].score > hits[-1].score


def test_gradio_panel_construction(tmp_path):
    from zara_bag_retrieval.data import ImageRecord
    from zara_bag_retrieval.ui import build_demo

    q = tmp_path / 'query' / 'img 001' / 'q.png'
    g = tmp_path / 'gallery' / 'img 001' / 'g.png'
    for item in (q, g):
        item.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (10, 10), 'blue').save(item)

    class FakeStore:
        query_records = [ImageRecord('img 001', q, 'img 001/q.png')]
        gallery_records = [ImageRecord('img 001', g, 'img 001/g.png')]
    class FakeService:
        store = FakeStore()
        available = ['dinov2']
        query_by_key = {'img 001/q.png': FakeStore.query_records[0]}
        def get_query_image_path(self, key):
            return str(q)
    demo = build_demo(FakeService())
    assert len(demo.blocks) >= 15
