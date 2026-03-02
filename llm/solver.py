"""
Solver - Main Answer Generator

Generates comprehensive answers based on retrieved context.
Supports both streaming and non-streaming modes.
"""

from typing import AsyncGenerator, Dict, Any, Optional, List
from llm.manager import LLMManager
from utils.logger import get_logger

logger = get_logger(__name__)


class Solver:
    """
    Answer generation module with streaming support.

    Features:
    - Context-aware generation
    - Streaming token output
    - Feedback-based refinement
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize Solver.

        Args:
            llm_manager: LLM manager instance
        """
        self.llm = llm_manager

        logger.info("Solver initialized")

    async def generate(
        self,
        query: str,
        context: str,
        feedback: Optional[List[str]] = None,
        prompt_override: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate answer (non-streaming).

        Args:
            query: User query
            context: Retrieved context
            feedback: Optional feedback from verifier

        Returns:
            Generated answer
        """
        logger.info("Generating answer", query=query[:100], context_length=len(context))

        if prompt_override:
            return await self.llm.generate(
                prompt_override,
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Build prompt
        variables = {
            "query": query,
            "context": context
        }

        # Add feedback if provided
        if feedback:
            feedback_text = "\n".join(f"- {f}" for f in feedback)
            variables["feedback"] = f"\n\n## Feedback to Address\n{feedback_text}"
        else:
            variables["feedback"] = ""

        # Generate
        answer = await self.llm.generate_with_template(
            "solver_system",
            variables,
            temperature=temperature,
            max_tokens=max_tokens
        )

        logger.info("Answer generated", length=len(answer))

        return answer

    async def generate_stream(
        self,
        query: str,
        context: str,
        feedback: Optional[List[str]] = None,
        prompt_override: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 20000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate answer with streaming.

        Args:
            query: User query
            context: Retrieved context
            feedback: Optional feedback

        Yields:
            Event dictionaries: {"type": "token", "data": "word"}
        """
        logger.info("Starting streaming generation", query=query[:100])

        if prompt_override:
            async for token in self.llm.generate_stream(
                prompt_override,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield {"type": "token", "data": token}
            return

        # Build prompt
        variables = {
            "query": query,
            "context": context
        }

        if feedback:
            feedback_text = "\n".join(f"- {f}" for f in feedback)
            variables["feedback"] = f"\n\n## Feedback to Address\n{feedback_text}"
        else:
            variables["feedback"] = ""

        # Stream tokens
        async for token in self.llm.generate_stream_with_template(
            "solver_system",
            variables,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            yield {"type": "token", "data": token}

    async def generate_general_chat(
        self,
        query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate response for general chat (no RAG context).

        Args:
            query: User query

        Yields:
            Token events
        """
        logger.info("Generating general chat response", query=query[:100])

        # Simple prompt without scientific context
        prompt = f"""You are an expert AI Assistant specializing in **Synthetic Lethality (SL) and Precision Oncology**.

The user has initiated a general conversation or greeting: "{query}"

Your Task:
1. Answer the user's input politely and briefly.
2. **Crucially**, steer the conversation back to your expertise. Invite the user to ask about Synthetic Lethality, cancer targets, drug mechanisms, or recent papers.

Example:
User: "Hello"
AI: "Hello! I'm ready to help. I specialize in analyzing Synthetic Lethality targets for cancer therapy. Do you have a specific gene or pathway you'd like to investigate today?"
"""

        async for token in self.llm.generate_stream(prompt, temperature=0.3, max_tokens=200):
            yield {"type": "token", "data": token}
