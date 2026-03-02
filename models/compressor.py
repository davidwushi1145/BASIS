"""
Context Compressor Model

Compresses long contexts while preserving key information.
Supports:
- Query-aware semantic sentence filtering (ev1 parity path)
- LLM-based compression
- Abstractive compression (Seq2Seq model)
- Extractive heuristic fallback
"""

from typing import List, Optional, Any, Dict
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except Exception:
    nltk = None
    sent_tokenize = None
    NLTK_AVAILABLE = False


# LLM compression prompt
LLM_COMPRESSION_PROMPT = """You are a biomedical research assistant. Compress the following text while preserving:
1. Key scientific claims and findings
2. Gene/protein names and relationships
3. Drug names and mechanisms
4. Quantitative results

Reduce to approximately {target_words} words. Keep essential information, remove redundancy.

Text to compress:
{text}

Compressed text:"""


class ContextCompressor:
    """
    Context compression for token budget management.

    Supports multiple compression strategies:
    - sentence_filter: query-aware semantic sentence filtering
    - hybrid: sentence_filter + optional LLM second-pass
    - llm: direct LLM compression
    - abstractive: Seq2Seq summarization model
    - extractive: heuristic sentence scoring fallback
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        max_input_length: int = 1024,
        max_output_length: int = 256,
        llm_client: Any = None,
        llm_model_name: str = None,
        embedder: Any = None,
        strategy: Optional[str] = None,
        ratio: Optional[float] = None,
        context_window: int = 1
    ):
        """
        Initialize Context Compressor.

        Args:
            model_path: Path to Seq2Seq summarization model
            device: Torch device
            max_input_length: Maximum input token length
            max_output_length: Maximum output token length
            llm_client: AsyncOpenAI client for LLM compression
            llm_model_name: Model name for LLM compression
            embedder: Embedder for query-aware semantic filtering
            strategy: Default compression strategy
            ratio: Compression ratio for sentence_filter/hybrid
            context_window: Keep neighboring sentences around top matches
        """
        self.model_path = model_path or getattr(settings, 'COMPRESSOR_MODEL_PATH', None)
        self.device = device or settings.DEVICE
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.llm_client = llm_client
        self.llm_model_name = llm_model_name or settings.SOLVER_MODEL_NAME
        self.embedder = embedder
        self.strategy = strategy or settings.COMPRESSION_STRATEGY
        self.ratio = ratio if ratio is not None else settings.COMPRESSION_RATIO
        self.context_window = max(0, int(context_window))

        self.model = None
        self.tokenizer = None
        self._nltk_sentence_split_ready = False
        self._init_nltk_sentence_splitter()

        if self.model_path:
            self._load_model()
        else:
            logger.info("Compressor model path not configured, semantic/LLM/extractive mode available")

        logger.info(
            "ContextCompressor initialized",
            strategy=self.strategy,
            ratio=self.ratio,
            context_window=self.context_window,
            has_embedder=bool(self.embedder),
            has_llm_client=bool(self.llm_client)
        )

    def set_llm_client(self, client: Any, model_name: str = None):
        """Set LLM client for compression."""
        self.llm_client = client
        if model_name:
            self.llm_model_name = model_name
        logger.info("LLM client set for compression", model=self.llm_model_name)

    def _init_nltk_sentence_splitter(self):
        """Enable NLTK sentence splitting only if tokenizer resources already exist."""
        if not NLTK_AVAILABLE:
            return

        try:
            nltk.data.find("tokenizers/punkt_tab")
            self._nltk_sentence_split_ready = True
            return
        except LookupError:
            pass

        try:
            nltk.data.find("tokenizers/punkt")
            self._nltk_sentence_split_ready = True
        except LookupError:
            self._nltk_sentence_split_ready = False
            logger.debug("NLTK punkt resource not found; using regex sentence split fallback")

    def _load_model(self):
        """Load the Seq2Seq compression model."""
        try:
            logger.info("Loading Context Compressor", path=self.model_path, device=self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            self.model.eval()

            logger.info("Context Compressor loaded successfully")
        except Exception as e:
            logger.error("Failed to load compressor model", error=str(e))
            self.model = None
            self.tokenizer = None

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with NLTK if available, else regex fallback."""
        if not text:
            return []

        if self._nltk_sentence_split_ready and sent_tokenize:
            try:
                return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 5]
            except Exception:
                pass

        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _filter_sentences_with_window(self, sentences: List[str], scores: torch.Tensor) -> str:
        """
        Keep top-scoring sentences and neighboring context windows.

        This matches ev1 behavior: select top semantic hits and include adjacent
        sentences to avoid losing local context.
        """
        if not sentences:
            return ""
        if len(sentences) <= 3:
            return " ".join(sentences)

        num_sentences = len(sentences)
        topk = max(1, int(num_sentences * self.ratio))
        topk = min(topk, num_sentences)

        _, top_indices = torch.topk(scores, k=topk)
        top_indices = top_indices.cpu().tolist()

        indices_to_keep = set()
        for idx in top_indices:
            start = max(0, idx - self.context_window)
            end = min(num_sentences - 1, idx + self.context_window)
            for i in range(start, end + 1):
                indices_to_keep.add(i)

        sorted_indices = sorted(indices_to_keep)
        return " ".join(sentences[i] for i in sorted_indices)

    def _query_semantic_compress(
        self,
        query: str,
        text: str,
        target_tokens: int
    ) -> Optional[str]:
        """Semantic compression using query-vs-sentence embedding similarity."""
        if not query or not text or not self.embedder:
            return None

        sentences = self._split_into_sentences(text)
        if not sentences:
            return None

        try:
            query_emb = self.embedder.encode(
                [query],
                is_query=True,
                task_instruction="Retrieve relevant biomedical evidence for the query"
            )
            sent_embs = self.embedder.encode(sentences)
            if query_emb.device != sent_embs.device:
                query_emb = query_emb.to(sent_embs.device)

            scores = torch.mm(query_emb, sent_embs.T).squeeze(0)
            compressed = self._filter_sentences_with_window(sentences, scores)
            if not compressed:
                return None

            words = compressed.split()
            if len(words) > target_tokens:
                compressed = " ".join(words[:target_tokens])
            return compressed

        except Exception as e:
            logger.warning("Semantic compression failed", error=str(e))
            return None

    def compress(
        self,
        text: str,
        target_tokens: int = 256,
        preserve_entities: Optional[List[str]] = None,
        strategy: str = None,
        query: Optional[str] = None
    ) -> str:
        """
        Sync compression helper.

        Note:
            Prefer `compress_async()` in async pipelines.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Avoid run_until_complete inside running event loop.
                return self._extractive_compress(text, target_tokens, preserve_entities)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.compress_async(
                text=text,
                target_tokens=target_tokens,
                preserve_entities=preserve_entities,
                strategy=strategy,
                query=query
            )
        )

    async def compress_async(
        self,
        text: str,
        target_tokens: int = 256,
        preserve_entities: Optional[List[str]] = None,
        strategy: str = None,
        query: Optional[str] = None
    ) -> str:
        """
        Async context compression.

        Args:
            text: Input text to compress
            target_tokens: Target output token count
            preserve_entities: Entities to preserve in fallback extractive mode
            strategy: Compression strategy
            query: User query for semantic compression
        """
        if not text:
            return ""

        if len(text.split()) <= target_tokens:
            return text

        strategy = strategy or self.strategy

        logger.debug(
            "Async compressing context",
            input_length=len(text),
            target_tokens=target_tokens,
            strategy=strategy,
            query_aware=bool(query and self.embedder)
        )

        if strategy in {"sentence_filter", "hybrid"}:
            semantic = self._query_semantic_compress(query or "", text, target_tokens)
            if semantic:
                if (
                    strategy == "hybrid"
                    and self.llm_client
                    and settings.USE_LLM_COMPRESSION
                    and len(semantic.split()) > max(80, target_tokens // 2)
                ):
                    try:
                        return await self._llm_compress(semantic, target_tokens)
                    except Exception as e:
                        logger.warning("Hybrid LLM second-pass failed", error=str(e))
                return semantic

        if strategy == "llm" and self.llm_client and settings.USE_LLM_COMPRESSION:
            try:
                return await self._llm_compress(text, target_tokens)
            except Exception as e:
                logger.warning("LLM compression failed", error=str(e))

        if strategy == "abstractive" and self.model and self.tokenizer:
            try:
                return self._abstractive_compress(text, target_tokens)
            except Exception as e:
                logger.warning("Abstractive compression failed", error=str(e))

        return self._extractive_compress(text, target_tokens, preserve_entities)

    async def compress_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch chunk compression (ev1-aligned API).

        Args:
            query: User query
            chunks: List of chunk dicts with at least `content`
            strategy: Optional strategy override

        Returns:
            New chunk list with compressed content and length metadata
        """
        if not chunks:
            return chunks

        strategy = strategy or self.strategy
        if strategy == "none":
            return chunks

        use_semantic_batch = (
            strategy in {"sentence_filter", "hybrid"}
            and bool(query)
            and bool(self.embedder)
        )

        all_sentences: List[str] = []
        chunk_map: List[Dict[str, Any]] = []

        for chunk in chunks:
            content = str((chunk or {}).get("content", "") or "")
            if use_semantic_batch:
                sentences = self._split_into_sentences(content)
                start = len(all_sentences)
                all_sentences.extend(sentences)
                end = len(all_sentences)
                chunk_map.append(
                    {
                        "chunk": chunk,
                        "content": content,
                        "sentences": sentences,
                        "start": start,
                        "end": end
                    }
                )
            else:
                chunk_map.append(
                    {
                        "chunk": chunk,
                        "content": content,
                        "sentences": [],
                        "start": 0,
                        "end": 0
                    }
                )

        all_scores = None
        if use_semantic_batch and all_sentences:
            try:
                query_emb = self.embedder.encode(
                    [query],
                    is_query=True,
                    task_instruction="Retrieve relevant biomedical evidence for the query"
                )
                sent_embs = self.embedder.encode(all_sentences)
                if query_emb.device != sent_embs.device:
                    query_emb = query_emb.to(sent_embs.device)
                all_scores = torch.mm(query_emb, sent_embs.T).squeeze(0)
            except Exception as e:
                logger.warning("Batch semantic compression unavailable, falling back", error=str(e))
                use_semantic_batch = False
                all_scores = None

        compressed_chunks: List[Dict[str, Any]] = []
        for entry in chunk_map:
            chunk = entry["chunk"]
            content = entry["content"]
            target_tokens = max(64, int(max(1, len(content.split())) * self.ratio))
            target_tokens = min(target_tokens, 600)
            compressed_content = content

            if use_semantic_batch and all_scores is not None and entry["start"] < entry["end"]:
                local_scores = all_scores[entry["start"]:entry["end"]]
                compressed_content = self._filter_sentences_with_window(entry["sentences"], local_scores)
                if len(compressed_content.split()) > target_tokens:
                    compressed_content = " ".join(compressed_content.split()[:target_tokens])

                if (
                    strategy == "hybrid"
                    and self.llm_client
                    and settings.USE_LLM_COMPRESSION
                    and len(compressed_content.split()) > max(80, target_tokens // 2)
                ):
                    try:
                        compressed_content = await self._llm_compress(compressed_content, target_tokens)
                    except Exception as e:
                        logger.warning("Chunk hybrid LLM pass failed", error=str(e))
            elif content:
                fallback_strategy = strategy
                if strategy in {"sentence_filter", "hybrid"}:
                    fallback_strategy = "extractive"
                compressed_content = await self.compress_async(
                    text=content,
                    target_tokens=target_tokens,
                    strategy=fallback_strategy,
                    query=query
                )

            new_chunk = dict(chunk)
            new_chunk["content"] = compressed_content
            new_chunk["original_length"] = len(content)
            new_chunk["compressed_length"] = len(compressed_content)
            compressed_chunks.append(new_chunk)

        return compressed_chunks

    async def _llm_compress(self, text: str, target_tokens: int) -> str:
        """
        LLM-based intelligent compression.
        """
        import asyncio

        target_words = int(target_tokens * 0.75)
        max_input_chars = 8000
        truncated_text = text[:max_input_chars] if len(text) > max_input_chars else text

        prompt = LLM_COMPRESSION_PROMPT.format(
            target_words=target_words,
            text=truncated_text
        )

        try:
            response = await asyncio.wait_for(
                self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=target_tokens + 50
                ),
                timeout=30.0
            )

            compressed = response.choices[0].message.content.strip()
            logger.debug(
                "LLM compression completed",
                input_length=len(text),
                output_length=len(compressed),
                ratio=(len(compressed) / len(text)) if text else 0
            )
            return compressed

        except asyncio.TimeoutError:
            logger.warning("LLM compression timeout")
            raise
        except Exception as e:
            logger.error("LLM compression error", error=str(e))
            raise

    def _abstractive_compress(self, text: str, target_tokens: int) -> str:
        """Abstractive compression using transformer model."""
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=min(target_tokens, self.max_output_length),
                min_length=min(target_tokens // 2, 50),
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        compressed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug("Abstractive compression completed", output_length=len(compressed))
        return compressed

    def _extractive_compress(
        self,
        text: str,
        target_tokens: int,
        preserve_entities: Optional[List[str]] = None
    ) -> str:
        """Heuristic extractive compression fallback."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s for s in sentences if s]

        if not sentences:
            return text[:target_tokens * 4]

        scored_sentences = []
        preserve_set = set(e.upper() for e in (preserve_entities or []))

        for i, sent in enumerate(sentences):
            score = 0.0

            if i < 3:
                score += 2 - (i * 0.5)

            if preserve_set:
                sent_upper = sent.upper()
                for entity in preserve_set:
                    if entity in sent_upper:
                        score += 3

            biomedical_keywords = [
                "gene", "protein", "mutation", "cancer", "tumor",
                "synthetic lethal", "inhibitor", "pathway", "expression",
                "crispr", "knockout", "dependency", "essentiality"
            ]
            sent_lower = sent.lower()
            for keyword in biomedical_keywords:
                if keyword in sent_lower:
                    score += 1

            if re.search(r"\d+\.?\d*%|\d+\.?\d*-fold|p\s*[<>=]\s*\d", sent):
                score += 2

            scored_sentences.append((score, i, sent))

        scored_sentences.sort(key=lambda x: (-x[0], x[1]))

        selected = []
        current_tokens = 0
        for _, idx, sent in scored_sentences:
            sent_tokens = len(sent.split())
            if current_tokens + sent_tokens <= target_tokens:
                selected.append((idx, sent))
                current_tokens += sent_tokens
            if current_tokens >= target_tokens:
                break

        selected.sort(key=lambda x: x[0])
        compressed = " ".join(sent for _, sent in selected)

        logger.debug(
            "Extractive compression completed",
            num_sentences_selected=len(selected),
            output_length=len(compressed)
        )

        return compressed

    def compress_batch(
        self,
        texts: List[str],
        target_tokens_each: int = 256,
        preserve_entities: Optional[List[str]] = None
    ) -> List[str]:
        """Compress multiple texts (sync fallback path)."""
        return [
            self.compress(text, target_tokens_each, preserve_entities)
            for text in texts
        ]

    def is_available(self) -> bool:
        """Check if abstractive compression is available."""
        return self.model is not None
