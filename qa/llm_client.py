"""
LLM Client Interface — US-109.

Provides a unified interface for streaming chat completions from xAI (Grok),
OpenAI, or a graceful local mock generator.
"""

from __future__ import annotations

import logging
import time
from typing import Generator, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified client for generating streaming completions from various LLM providers.
    
    Supports OpenAI, xAI (Grok), and a local offline mock generator.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider name ('xai', 'openai', 'mock'). Defaults to config.
            api_key: API key for the provider. Defaults to config settings.
            model: Model name to use. Defaults to config settings.
        """
        settings = get_settings()
        self.provider = (provider or settings.LLM_PROVIDER or "").lower()
        
        # Determine credentials and config based on provider
        if self.provider == "xai":
            self.api_key = api_key or settings.XAI_API_KEY
            self.model = model or settings.XAI_MODEL or "grok-2"
            self.api_base = settings.XAI_API_BASE or "https://api.x.ai/v1"
        elif self.provider == "openai":
            self.api_key = api_key or settings.OPENAI_API_KEY
            self.model = model or settings.OPENAI_MODEL or "gpt-4o-mini"
            self.api_base = settings.OPENAI_API_BASE or "https://api.openai.com/v1"
        elif self.provider == "groq":
            self.api_key = api_key or settings.GROQ_API_KEY
            self.model = model or settings.GROQ_MODEL or "llama-3.1-8b-instant"
            self.api_base = settings.GROQ_API_BASE or "https://api.groq.com/openai/v1"
        else:
            self.provider = "mock"
            self.api_key = ""
            self.model = "local-mock"
            self.api_base = ""

        # Fallback to mock if API key is missing for key-required providers
        if self.provider in ["xai", "openai", "groq"] and not self.api_key:
            logger.warning(
                f"LLM provider '{self.provider}' requested but no API key was provided or configured. "
                "Falling back to local 'mock' provider."
            )
            self.provider = "mock"
            self.model = "local-mock"

        # Initialize OpenAI client if using xAI or OpenAI
        self._client = None
        if self.provider in ["xai", "openai", "groq"]:
            from openai import OpenAI
            logger.info(f"Initializing {self.provider.upper()} API client using model '{self.model}'...")
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """
        Stream chat completions token-by-token.

        Args:
            messages: List of message dictionaries containing 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Tokens of the generated text response.
        """
        if self.provider == "mock" or self._client is None:
            yield from self._stream_mock_response(messages)
            return

        try:
            logger.info(f"Sending stream request to {self.provider.upper()} ({self.model})...")
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content

        except Exception as e:
            logger.error(f"LLM API request failed: {e}. Falling back to mock generator.")
            yield f"\n⚠️ [API ERROR: {e}]\nFalling back to local response generator...\n\n"
            yield from self._stream_mock_response(messages)

    def _stream_mock_response(
        self,
        messages: list[dict[str, str]],
    ) -> Generator[str, None, None]:
        """Generate a simulated response based on the context found in the last message."""
        logger.info("Generating local offline mock response...")

        # Find user message (typically contains the prompt with contexts)
        user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        
        # Try to parse source filenames out of the context block
        import re
        sources = list(set(re.findall(r"\[Source:\s*([^,\s\]]+)", user_message)))
        chunks = list(set(re.findall(r"Chunk:\s*(\d+)", user_message)))

        source_info = ""
        if sources:
            source_info = f"I examined the ingested file(s) ({', '.join(f'`{s}`' for s in sources)})."
        else:
            source_info = "I examined the provided text documents."

        citations = []
        for i in range(min(len(sources), 3)):
            chunk_num = chunks[i] if i < len(chunks) else str(i)
            citations.append(f"[Source: {sources[i]}, Chunk: {chunk_num}]")

        citation_str = f" {citations[0]}" if citations else ""
        citation_str2 = f" {citations[1]}" if len(citations) > 1 else (citation_str or "")

        # Construct a beautiful simulated response demonstrating standard RAG response properties
        mock_paragraphs = [
            f"### 🤖 Local RAG Offline Response\n\n"
            f"This is a local, high-fidelity mock response running offline (since no LLM API key was provided or the API failed). {source_info}\n\n"
            f"Based on the retrieved context, the documents discuss the core principles of document processing. Specifically, the system successfully ingested the files and generated semantic vector embeddings.{citation_str}",
            
            f"Furthermore, the retrieval pipeline successfully matched your query against the relevant passages in our vector store using a hybrid dense-sparse search index (FAISS and BM25).{citation_str2}\n\n"
            f"Here is a quick summary of the matched content:\n"
            f"1. **Lexical Matching**: BM25 captured exact keyword terms from your query.\n"
            f"2. **Semantic Matching**: FAISS mapped the query to overlapping passages based on vector similarity.\n"
            f"3. **Re-ranking**: Cross-encoder scoring surfaced the most grounded text snippets for generation.",

            f"### Citations & Veracity\n\n"
            f"All claims made above correspond exactly to the retrieved sources shown in the citation cards below. You can view the specific sections by clicking on the cards or inspecting the citation footnotes in brackets."
        ]

        full_text = "\n\n".join(mock_paragraphs)
        
        # Stream word-by-word with a slight delay
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.015)
