"""
Token Counter

Counts tokens for budget allocation using tiktoken.
"""

import tiktoken
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class TokenCounter:
    """
    Token counting utility using tiktoken.

    Uses cl100k_base encoding (GPT-4/GPT-3.5 compatible).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize Token Counter.

        Args:
            encoding_name: Tiktoken encoding name
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.info("TokenCounter initialized", encoding=encoding_name)
        except Exception as e:
            logger.warning(
                "Tiktoken not available, using approximation",
                error=str(e)
            )
            self.encoding = None

    async def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Approximation: ~4 chars per token
            return len(text) // 4

    async def truncate(
        self,
        text: str,
        max_tokens: int,
        from_end: bool = False
    ) -> str:
        """
        Truncate text to fit token budget.

        Args:
            text: Input text
            max_tokens: Maximum tokens
            from_end: If True, truncate from end; else from start

        Returns:
            Truncated text
        """
        if not self.encoding:
            # Approximation
            max_chars = max_tokens * 4
            if from_end:
                return text[-max_chars:]
            else:
                return text[:max_chars]

        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        if from_end:
            truncated_tokens = tokens[-max_tokens:]
        else:
            truncated_tokens = tokens[:max_tokens]

        return self.encoding.decode(truncated_tokens)
