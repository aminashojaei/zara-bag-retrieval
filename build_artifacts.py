import argparse
import csv
import gc
import json
from pathlib import Path
from PIL import Image

import numpy as np
from zara_bag_retrieval.config import DEFAULT_DATASET, DEFAULT_RESULTS, MODEL_IDS
from zara_bag_retrieval.data import scan_dataset_role
from zara_bag_retrieval.encoders import EncoderPool


def encode_records(pool, key, records, batch_size):
    vectors = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        images = []
        for r in batch:
            with Image.open(r.path) as src:
                images.append(src.copy())
        vectors.append(pool.images_embeddings(key, images))
        print(f"{key}: embedded {start+len(batch)}/{len(records)}", end="\r", flush=True)
    print()
    return np.ascontiguousarray(np.vstack(vectors).astype(np.float32))


def write_inventory(path, records, role):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['product_id','path'])
        writer.writeheader()
        writer.writerows({'product_id': r.product_id, 'path': f'{role}/{r.relative_key}'} for r in records)


def main():
    p = argparse.ArgumentParser(description='Build model indexes')
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--out', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--models', nargs='+', choices=tuple(MODEL_IDS), default=list(MODEL_IDS))
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--force', action='store_true', help='Overwrite existing model artifact folders')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be >= 1')
    queries = scan_dataset_role(args.dataset, 'query')
    gallery = scan_dataset_role(args.dataset, 'gallery')
    if not queries or not gallery:
        p.error('Dataset must contain both query/ and gallery/ images')
    if not {r.product_id for r in queries}.issubset({r.product_id for r in gallery}):
        p.error('Some query products are missing in the gallery')
    import faiss
    pool = EncoderPool(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        folder = args.out / key
        if folder.exists() and any(folder.iterdir()) and not args.force:
            print(f'Skipping {key}: {folder} already contains artifacts. Use --force to rebuild.')
            continue
        folder.mkdir(parents=True, exist_ok=True)
        print(f'\n=== Building {key} ===')
        try:
            qvec = encode_records(pool, key, queries, args.batch_size)
            gvec = encode_records(pool, key, gallery, args.batch_size)
            if qvec.shape[1] != gvec.shape[1]:
                raise ValueError('Query and gallery embedding dimensions differ')
            index = faiss.IndexFlatIP(gvec.shape[1])
            index.add(gvec)
            np.save(folder / 'query_embeddings.npy', qvec)
            np.save(folder / 'gallery_embeddings.npy', gvec)
            faiss.write_index(index, str(folder / 'gallery.faiss'))
            write_inventory(folder / 'query_inventory.csv', queries, 'query')
            write_inventory(folder / 'gallery_inventory.csv', gallery, 'gallery')
            (folder / 'build_info.json').write_text(json.dumps({
                'model': MODEL_IDS[key], 'query_images': len(queries),
                'gallery_images': len(gallery), 'embedding_dim': int(gvec.shape[1]),
                'normalized': True, 'index_type': 'IndexFlatIP'
            }, indent=2), encoding='utf-8')
            print(f'Saved {folder}')
        except Exception as e:
            print(f'FAILED {key}: {type(e).__name__}: {e}')
            print('Continue with the remaining models.')
        finally:
            gc.collect()


if __name__ == '__main__':
    main()
