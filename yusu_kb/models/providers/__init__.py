"""模型供应商管理：builtin 模板、业务层与进程内缓存。"""

from yusu_kb.models.providers.builtin import BUILTIN_PROVIDERS
from yusu_kb.models.providers.cache import (
    ModelCache,
    ModelInfo,
    model_cache,
    resolve_model_spec,
)
from yusu_kb.models.providers.service import (
    check_credential_status,
    create_provider_config,
    delete_provider_config,
    ensure_builtin_model_providers_in_db,
    fetch_remote_models,
    get_all_model_providers,
    get_model_provider_by_id,
    resolve_api_key,
    test_model_status_by_spec,
    update_provider_config,
)

__all__ = [
    "BUILTIN_PROVIDERS",
    "ModelCache",
    "ModelInfo",
    "check_credential_status",
    "create_provider_config",
    "delete_provider_config",
    "ensure_builtin_model_providers_in_db",
    "fetch_remote_models",
    "get_all_model_providers",
    "get_model_provider_by_id",
    "model_cache",
    "resolve_api_key",
    "resolve_model_spec",
    "test_model_status_by_spec",
    "update_provider_config",
]
