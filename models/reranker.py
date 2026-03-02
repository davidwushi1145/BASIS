"""
Qwen Reranker Model

Wrapper for Qwen3-Reranker-0.6B model for relevance scoring.
"""

import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class QwenReranker:
    """
    Qwen3 Reranker model wrapper.

    Uses yes/no token logits to compute relevance scores.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize Qwen Reranker.

        Args:
            model_path: Path to Qwen3-Reranker model
            device: Torch device
        """
        self.model_path = model_path or settings.RERANKER_PATH
        self.device = device or settings.DEVICE

        logger.info("Loading Qwen Reranker", path=self.model_path, device=self.device)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Special tokens
        self.token_yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.tokenizer.convert_tokens_to_ids("no")

        # Prefix and suffix templates
        self.prefix_tokens = self.tokenizer.encode(
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n",
            add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False
        )

        logger.info("Qwen Reranker loaded successfully")

    def compute_score(
        self,
        query: str,
        document: str,
        instruction: Optional[str] = None,
        domain: str = "synthetic_lethality"
    ) -> float:
        """
        Compute query-document relevance score.

        Args:
            query: User query
            document: Candidate document
            instruction: Custom instruction (auto-selected by domain if None)
            domain: Domain type ("synthetic_lethality" | "general")

        Returns:
            Relevance score [0.0, 1.0]
        """
        # Select instruction based on domain
        if instruction is None:
            if domain == "synthetic_lethality":
                instruction = """Given a query about Synthetic Lethality or Cancer Therapy, retrieve documents describing:
(1) Gene-gene synthetic lethal interactions or functional dependencies
(2) Small molecule inhibitors targeting SL partners or compensatory pathways
(3) Clinical trials exploiting SL vulnerabilities (e.g., PARP inhibitors in BRCA-deficient cancers)
(4) Mechanistic explanations of genetic dependencies, pathway rewiring, or drug resistance
Prioritize documents with gene names, drug mechanisms, essentiality scores, or experimental evidence.
Ignore general descriptions of genes without interaction details."""
            else:
                instruction = "Given a web search query, retrieve relevant passages that answer the query"

        logger.debug(
            "Computing reranker score",
            query=query[:100],
            document=document[:100],
            domain=domain
        )

        # Build input text
        raw_text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        input_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
        original_length = len(input_ids)

        # Truncate if necessary
        max_length = 8192 - len(self.prefix_tokens) - len(self.suffix_tokens)
        input_ids = input_ids[:max_length]

        # Add prefix and suffix
        final_input_ids = self.prefix_tokens + input_ids + self.suffix_tokens
        input_tensor = torch.tensor([final_input_ids], device=self.device)

        if original_length > max_length:
            logger.debug(
                "Input truncated",
                original_length=original_length,
                final_length=len(final_input_ids)
            )

        # Compute logits
        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits[0, -1]
            yes_logit = logits[self.token_yes_id]
            no_logit = logits[self.token_no_id]

            # Softmax to get probability
            score = torch.exp(yes_logit) / (torch.exp(yes_logit) + torch.exp(no_logit))

        score_value = score.item()

        logger.debug(
            "Reranker score computed",
            score=score_value,
            yes_logit=yes_logit.item(),
            no_logit=no_logit.item()
        )

        return score_value

    def rerank_batch(
        self,
        query: str,
        documents: list[str],
        domain: str = "synthetic_lethality"
    ) -> list[float]:
        """
        Rerank a batch of documents.

        Args:
            query: User query
            documents: List of candidate documents
            domain: Domain type

        Returns:
            List of relevance scores
        """
        scores = []
        for doc in documents:
            score = self.compute_score(query, doc, domain=domain)
            scores.append(score)

        logger.debug(
            "Batch reranking completed",
            num_docs=len(documents),
            avg_score=sum(scores) / len(scores) if scores else 0.0
        )

        return scores
