"""
embedder.py

Wraps InLegalBERT (law-ai/InLegalBERT) as a sentence encoder using
mean pooling over the last hidden state.

InLegalBERT is a masked-LM BERT model, not a sentence-transformer,
so we load it via HuggingFace and apply mean pooling manually.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

MODEL_NAME = "law-ai/InLegalBERT"
MAX_LENGTH = 512


class LegalEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Embedder ready.")

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool token embeddings, ignoring padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    def encode(self, texts: list[str], batch_size: int = 32, normalize: bool = True) -> list[list[float]]:
        """
        Encode a list of texts into embeddings.
        Returns list of float lists (compatible with ChromaDB).
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]
