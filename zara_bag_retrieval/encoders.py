import gc
from threading import RLock

import numpy as np
from PIL import Image, ImageOps

from .config import MODEL_IDS


class EncoderPool:
    def __init__(self, device: str = "auto"):
        self.device_setting = device
        self.current_key: str | None = None
        self.model = None
        self.processor = None
        self.lock = RLock()

    def _device(self) -> str:
        import torch
        if self.device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return self.device_setting

    def _load(self, key: str):
        if key == self.current_key and self.model is not None:
            return
        if key not in MODEL_IDS:
            raise ValueError(f"Unknown encoder: {key}")
        self.model = None
        self.processor = None
        self.current_key = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from transformers import AutoModel, AutoImageProcessor, AutoProcessor
        model_id = MODEL_IDS[key]
        processor_cls = AutoProcessor if key == "siglip2" else AutoImageProcessor
        self.processor = processor_cls.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self._device()).eval()
        self.current_key = key

    def images_embeddings(self, key: str, images: list[Image.Image]) -> np.ndarray:
        import torch
        if not images:
            raise ValueError("At least one image is required")
        with self.lock:
            self._load(key)
            rgb = [ImageOps.exif_transpose(im).convert("RGB") for im in images]
            inputs = self.processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self._device()) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                if key == "siglip2":
                    output = self.model.get_image_features(**inputs)
                    if isinstance(output, torch.Tensor):
                        features = output
                    elif getattr(output, "image_embeds", None) is not None:
                        features = output.image_embeds
                    elif getattr(output, "pooler_output", None) is not None:
                        features = output.pooler_output
                    elif isinstance(output, (tuple, list)):
                        features = output[0]
                    else:
                        raise RuntimeError("Unexpected SigLIP2 image feature format")
                else:
                    output = self.model(**inputs)
                    features = output.last_hidden_state[:, 0, :]
            if features.ndim != 2 or features.shape[0] != len(images):
                raise RuntimeError(f"Unexpected encoder output shape: {features.shape}")
            vecs = features.detach().float().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            if not np.isfinite(vecs).all() or (norms < 1e-12).any():
                raise ValueError("Non-finite or empty model embeddings")
            return np.ascontiguousarray(vecs / norms)

    def image_embedding(self, key: str, image: Image.Image) -> np.ndarray:
        if image is None:
            raise ValueError("Please upload an image")
        return self.images_embeddings(key, [image])[0]
