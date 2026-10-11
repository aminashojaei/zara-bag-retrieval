from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_IDS = {
    "dinov2": "facebook/dinov2-small",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "siglip2": "google/siglip2-base-patch16-224",
}
MODEL_LABELS = {
    "dinov2": "DINOv2 Small",
    "dinov3": "DINOv3 Small",
    "siglip2": "SigLIP2 Base",
}
REQUIRED_GALLERY_FILES = ("gallery.faiss", "gallery_embeddings.npy", "gallery_inventory.csv")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_DATASET = PROJECT_ROOT / "data"
DEFAULT_RESULTS = PROJECT_ROOT / "artifacts"
