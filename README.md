# Zara Bag Retrieval

Visual product retrieval with DINOv2, DINOv3, SigLIP2, FAISS and Gradio.

## Setup

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

Place the original product images inside `data/query/` and `data/gallery/`. Each product must use the same directory name in both locations.

```text
Zara_Bag_Retrieval/
├── app.py
├── build_artifacts.py
├── requirements.txt
├── data/
│   ├── query/
│   │   ├── img 001/
│   │   └── ...
│   └── gallery/
│       ├── img 001/
│       └── ...
├── artifacts/           (generated locally; not tracked by Git)
├── zara_bag_retrieval/
└── tests/
```

If you have existing model artifacts, copy the `dinov2`, `dinov3` and `siglip2` directories into `artifacts/`. The app validates that the image inventory matches the current dataset.

Otherwise, build the indexes once:

```bash
python build_artifacts.py --batch-size 2
```

DINOv3 requires access to its gated Hugging Face repository (`hf auth login`). If access is unavailable, build the other models with `--models dinov2 siglip2`.

Start the application from the project directory:

```bash
python app.py
```

Open http://127.0.0.1:7860.

## Search

- Select one of the saved query images, or upload a new image.
- Choose DINOv2, DINOv3 or SigLIP2.
- View the top three **different products**, with one representative image and similarity score per product.
- Saved queries reuse cached embeddings. Uploaded images are encoded on demand.

Original image files are not resized or rewritten by the application. Model preprocessing takes place in memory.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Repository data

The `data/` directory is intended for the project images. Check redistribution rights for Zara photography before publishing it. GitHub rejects individual files larger than 100 MB; use Git LFS when appropriate.

Generated indexes, embeddings and model weights are local runtime files and are excluded from Git. No search history or result files are saved by the web interface.


## Features

- Search by saved query image
- Search by uploaded image
- Search by text with SigLIP2
- Return top 3 distinct products
