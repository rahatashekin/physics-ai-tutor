from typing import Any
from tiktoken import get_encoding
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer


class OpenAITokenizerWrapper(BaseTokenizer):
    """Minimal wrapper for OpenAI's tiktoken tokenizer.

    Updated to match the new docling-core BaseTokenizer interface,
    which is now a Pydantic BaseModel with only 3 abstract methods:
      - count_tokens()
      - get_max_tokens()
      - get_tokenizer()
    """

    # Pydantic model fields (use model_config to allow arbitrary types)
    model_config = {"arbitrary_types_allowed": True}

    _model_name: str = "cl100k_base"
    _max_length: int = 8191
    _tokenizer: Any = None

    def __init__(
        self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs
    ):
        super().__init__(**kwargs)
        # Store as private attributes (not Pydantic fields)
        object.__setattr__(self, "_model_name", model_name)
        object.__setattr__(self, "_max_length", max_length)
        object.__setattr__(self, "_tokenizer", get_encoding(model_name))

    def count_tokens(self, text: str) -> int:
        """Count tokens for the given text."""
        return len(self._tokenizer.encode(text))

    def get_max_tokens(self) -> int:
        """Return maximum sequence length."""
        return self._max_length

    def get_tokenizer(self) -> Any:
        """Return underlying tiktoken encoding object."""
        return self._tokenizer
