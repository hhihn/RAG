from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_torch_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.device = get_torch_device()
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)
