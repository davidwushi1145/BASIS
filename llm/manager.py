"""
LLM Manager - Unified Interface for All LLM Calls

Handles prompt loading, model calls, and response parsing.
"""

import json
from typing import Dict, Any, Optional, AsyncGenerator
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI
import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PromptLoader:
    """Loads and renders Jinja2 prompt templates"""

    def __init__(self, template_dir: str = "config/prompts"):
        """
        Initialize prompt loader.

        Args:
            template_dir: Directory containing .j2 template files
        """
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

        logger.info("PromptLoader initialized", template_dir=str(self.template_dir))

    def render(self, template_name: str, **kwargs) -> str:
        """
        Render a prompt template.

        Args:
            template_name: Template filename (without .j2 extension)
            **kwargs: Variables to pass to template

        Returns:
            Rendered prompt string
        """
        template_file = f"{template_name}.j2"
        template = self.env.get_template(template_file)
        rendered = template.render(**kwargs)

        logger.debug(
            "Prompt rendered",
            template=template_name,
            length=len(rendered)
        )

        return rendered


class LLMManager:
    """
    Unified LLM interface for all model calls.

    Handles:
    - Prompt rendering
    - API calls (AsyncOpenAI)
    - Streaming responses
    - Error handling and retries
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        proxy_url: Optional[str] = None,
        default_model: Optional[str] = None,
        template_dir: Optional[str] = None
    ):
        """
        Initialize LLM Manager.

        Args:
            api_key: OpenAI-compatible API key
            api_base: API base URL
            proxy_url: HTTP proxy URL
            default_model: Default model name
            template_dir: Prompt template directory
        """
        self.api_key = api_key or settings.API_KEY
        self.api_base = api_base or settings.SOLVER_API_BASE
        self.proxy_url = proxy_url or settings.PROXY_URL
        self.default_model = default_model or settings.SOLVER_MODEL_NAME

        # Create HTTP client with proxy
        http_client = None
        if self.proxy_url:
            http_client = httpx.AsyncClient(proxy=self.proxy_url)

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=http_client
        )

        # Initialize prompt loader
        self.prompt_loader = PromptLoader(template_dir or "config/prompts")

        logger.info(
            "LLMManager initialized",
            api_base=self.api_base,
            default_model=self.default_model,
            use_proxy=bool(self.proxy_url)
        )

    def get_client(self) -> AsyncOpenAI:
        """Expose underlying async OpenAI-compatible client."""
        return self.client

    async def close(self):
        """Close underlying HTTP resources if available."""
        try:
            await self.client.close()
        except Exception:
            # Client implementations may not expose close in all versions.
            pass

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> str:
        """
        Generate completion (non-streaming).

        Args:
            prompt: Input prompt
            model: Model name (default: settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters

        Returns:
            Generated text
        """
        model = model or self.default_model

        logger.debug(
            "Generating completion",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            content = response.choices[0].message.content.strip()

            logger.debug(
                "Completion generated",
                length=len(content),
                model=model
            )

            return content

        except Exception as e:
            logger.error("Generation failed", error=str(e), model=model)
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate completion with streaming.

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Token strings
        """
        model = model or self.default_model

        logger.debug("Starting streaming generation", model=model)

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("Streaming failed", error=str(e), model=model)
            raise

    async def generate_with_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate using a Jinja2 template.

        Args:
            template_name: Template file name (without .j2)
            variables: Template variables
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        prompt = self.prompt_loader.render(template_name, **variables)
        return await self.generate(prompt, **kwargs)

    async def generate_stream_with_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming using a template.

        Args:
            template_name: Template name
            variables: Template variables
            **kwargs: Generation parameters

        Yields:
            Token strings
        """
        prompt = self.prompt_loader.render(template_name, **variables)
        async for token in self.generate_stream(prompt, **kwargs):
            yield token

    async def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles markdown code blocks: ```json ... ```

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("JSON parsing failed", error=str(e), response=response[:200])
            raise ValueError(f"Invalid JSON response: {str(e)}")
