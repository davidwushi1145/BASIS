"""
Qwen Embedder Model

Wrapper for Qwen3-Embedding-0.6B model with async support.
"""

import torch
import torch.nn.functional as F
from typing import List, Union
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class QwenEmbedder:
    """
    Qwen3 Embedding model wrapper.

    Features:
    - Last token pooling
    - L2 normalization
    - Batch encoding
    - Query/Document mode
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize Qwen Embedder.

        Args:
            model_path: Path to Qwen3-Embedding model
            device: Torch device (cuda:0, cpu, etc.)
        """
        self.model_path = model_path or settings.EMBEDDING_PATH
        self.device = device or settings.DEVICE

        logger.info("Loading Qwen Embedder", path=self.model_path, device=self.device)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='left'
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        logger.info("Qwen Embedder loaded successfully")

    def _last_token_pool(
        self,
        last_hidden_states: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        Pool embeddings using last token.

        Args:
            last_hidden_states: Model outputs
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode(
        self,
        texts: List[str],
        is_query: bool = False,
        task_instruction: str = ""
    ) -> Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings
            is_query: If True, add query instruction
            task_instruction: Task-specific instruction

        Returns:
            Normalized embeddings tensor [batch_size, embedding_dim]
        """
        logger.debug(
            "Encoding texts",
            num_texts=len(texts),
            is_query=is_query,
            instruction=task_instruction[:50] if task_instruction else ""
        )

        # Prepare input texts
        if is_query:
            input_texts = [
                f'Instruct: {task_instruction}\nQuery:{q}' for q in texts
            ]
        else:
            input_texts = texts

        # Tokenize
        batch_dict = self.tokenizer(
            input_texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Encode
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)

        logger.debug(
            "Encoding completed",
            input_shape=batch_dict['input_ids'].shape,
            output_shape=embeddings.shape
        )

        return embeddings

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch encode and return as list of lists.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        embeddings = self.encode(texts, is_query=False)
        return embeddings.cpu().tolist()

    async def embed_async(self, text: str, is_query: bool = True) -> List[float]:
        """
        Async single text embedding (for compatibility).

        Args:
            text: Single text string
            is_query: If True, use query mode

        Returns:
            Single embedding vector
        """
        embeddings = self.encode(
            [text],
            is_query=is_query,
            task_instruction="Retrieve relevant documents for the query" if is_query else ""
        )
        return embeddings[0].cpu().tolist()

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        # Test with dummy input
        dummy_embedding = self.encode(["test"], is_query=False)
        return dummy_embedding.shape[1]
