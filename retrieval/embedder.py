"""
embedder.py

Uses sentence-transformers/all-mpnet-base-v2 for dense retrieval.

Why not InLegalBERT:
  InLegalBERT is a masked language model (MLM) trained for classification.
  Mean-pooling its hidden states gives poor sentence embeddings for retrieval
  because it was never trained with a similarity objective.

Why all-mpnet-base-v2:
  Trained with contrastive loss (multiple negatives ranking loss) specifically
  for semantic similarity and retrieval. Consistently outperforms BM25 on
  open-domain retrieval tasks out of the box.
"""

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class LegalEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Embedder ready.")

    def encode(self, texts: list[str], batch_size: int = 128, normalize: bool = True) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]
