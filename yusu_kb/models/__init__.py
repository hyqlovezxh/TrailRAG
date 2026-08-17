"""Self-contained model layer: embedding, reranker and chat clients."""

from yusu_kb.models.chat import (
    GeneralResponse,
    OpenAIChatAdapter,
    create_chat_model,
    test_chat_model_status,
)
from yusu_kb.models.embed import (
    BaseEmbeddingModel,
    EmbeddingFunc,
    OtherEmbedding,
    create_default_embedding_func,
    create_embedding_model,
    wrap_embedding_func_with_attrs,
)
from yusu_kb.models.rerank import (
    BaseReranker,
    DashscopeReranker,
    OpenAIReranker,
    close_cached_rerankers,
    create_reranker,
    get_cached_reranker,
)

__all__ = [
    "BaseEmbeddingModel",
    "BaseReranker",
    "DashscopeReranker",
    "EmbeddingFunc",
    "GeneralResponse",
    "OpenAIChatAdapter",
    "OpenAIReranker",
    "OtherEmbedding",
    "close_cached_rerankers",
    "create_chat_model",
    "create_default_embedding_func",
    "create_embedding_model",
    "create_reranker",
    "get_cached_reranker",
    "test_chat_model_status",
    "wrap_embedding_func_with_attrs",
]