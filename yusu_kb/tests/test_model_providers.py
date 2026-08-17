"""Model provider tests: env key resolution, remote model fetch, CRUD, cache.

Uses a throwaway SQLite DB (conftest ``engine`` fixture) + httpx.MockTransport;
no external services are contacted.
"""

from __future__ import annotations

import httpx
import pytest

from yusu_kb.models.providers import cache as cache_module
from yusu_kb.models.providers import service as svc
from yusu_kb.models.providers.builtin import BUILTIN_PROVIDERS
from yusu_kb.models.providers.cache import model_cache, resolve_model_spec
from yusu_kb.repositories import configure_repositories
from yusu_kb.storage.sqlite.models_knowledge import ModelProvider


def _provider(**overrides) -> ModelProvider:
    base = {
        "provider_id": "test-provider",
        "display_name": "Test Provider",
        "provider_type": "openai",
        "base_url": "https://provider.test/v1",
        "embedding_base_url": "https://provider.test/v1/embeddings",
        "rerank_base_url": "https://provider.test/v1/rerank",
        "api_key_env": "TEST_PROVIDER_KEY",
        "capabilities": ["chat", "embedding"],
        "enabled_models": [
            {"id": "chat-model", "type": "chat", "display_name": "Chat"},
            {"id": "embed-model", "type": "embedding", "display_name": "Embed", "dimension": 8},
        ],
        "headers_json": {},
        "extra_json": {},
        "is_enabled": True,
        "is_builtin": False,
    }
    base.update(overrides)
    return ModelProvider(**base)


@pytest.fixture()
def repos(engine):
    configure_repositories(engine)
    yield
    configure_repositories(None)


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Clear the module-level model cache between tests."""
    model_cache._cache = {}
    model_cache._cache_at = 0.0
    yield
    model_cache._cache = {}
    model_cache._cache_at = 0.0


# ============================================================ key resolution


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-env-value")
    assert svc.resolve_api_key(_provider()) == "sk-env-value"


def test_resolve_api_key_missing_env(monkeypatch):
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    assert svc.resolve_api_key(_provider()) is None


def test_resolve_api_key_without_env_name():
    provider = _provider(api_key_env=None)
    assert svc.resolve_api_key(provider) is None


def test_check_credential_status(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-ok")
    provider = _provider()
    assert svc.check_credential_status(provider) == "ok"
    assert svc.check_credential_status(_provider(is_enabled=False)) == "ok"

    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    assert svc.check_credential_status(provider) == "warning"
    assert svc.check_credential_status(_provider(api_key_env=None)) == "warning"


# ============================================================ remote models


def _remote_models_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "sub_type=embedding" in url:
        return httpx.Response(200, json={"data": [{"id": "embed-model", "object": "model"}]})
    if "sub_type=reranker" in url:
        return httpx.Response(200, json={"data": [{"id": "rerank-model"}]})
    # duplicate "chat-model" inside one endpoint: must be deduped by (id, type)
    return httpx.Response(200, json={"data": [{"id": "chat-model", "name": "Chat Model"}, {"id": "chat-model"}]})


async def test_fetch_remote_models_parses_and_dedupes():
    provider = _provider(
        base_url="https://provider.test/v1",
        embedding_base_url="https://provider.test/v1/embeddings",
        rerank_base_url="https://provider.test/v1/rerank",
        capabilities=["chat", "embedding", "rerank"],
        models_endpoint="models?sub_type=chat",
        embedding_models_endpoint="models?sub_type=embedding",
        rerank_models_endpoint="models?sub_type=reranker",
    )
    transport = httpx.MockTransport(_remote_models_handler)
    models = await svc.fetch_remote_models(provider, transport=transport)
    by_key = {(m["id"], m["type"]): m for m in models}
    assert len(models) == 3
    assert by_key[("chat-model", "chat")]["display_name"] == "Chat Model"
    assert by_key[("embed-model", "embedding")]["type"] == "embedding"
    assert by_key[("rerank-model", "rerank")]["type"] == "rerank"


async def test_fetch_remote_models_401_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    transport = httpx.MockTransport(handler)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await svc.fetch_remote_models(
            _provider(models_endpoint="models", embedding_models_endpoint="embeddings/models"),
            transport=transport,
        )
    assert "401" in str(exc_info.value)


# ================================================================ CRUD


async def test_provider_crud_roundtrip_and_idempotency(repos):
    data = {
        "provider_id": "crud-provider",
        "display_name": "CRUD Provider",
        "base_url": "https://crud.test/v1",
        "api_key_env": "CRUD_KEY",
        "capabilities": ["chat"],
        "enabled_models": [{"id": "c1", "type": "chat"}],
    }
    created = await svc.create_provider_config(data, username="tester")
    assert created.provider_id == "crud-provider"
    assert created.created_by == "tester"
    assert created.is_builtin is False

    with pytest.raises(ValueError, match="已存在"):
        await svc.create_provider_config(data, username="tester")

    fetched = await svc.get_model_provider_by_id("crud-provider")
    assert fetched is not None
    assert fetched.display_name == "CRUD Provider"
    assert len(await svc.get_all_model_providers()) == 1

    updated = await svc.update_provider_config("crud-provider", {"display_name": "改名"}, "tester")
    assert updated is not None
    assert updated.display_name == "改名"

    assert await svc.update_provider_config("missing", {"display_name": "x"}, "tester") is None

    assert await svc.delete_provider_config("crud-provider") is True
    assert await svc.delete_provider_config("crud-provider") is False


async def test_update_provider_models_capability_validation(repos):
    await svc.create_provider_config(
        {
            "provider_id": "cap-provider",
            "display_name": "Cap",
            "base_url": "https://cap.test/v1",
            "capabilities": ["chat"],
            "enabled_models": [{"id": "c1", "type": "chat"}],
        },
        username="tester",
    )
    with pytest.raises(ValueError, match="不在 provider 能力"):
        await svc.update_provider_config(
            "cap-provider",
            {"enabled_models": [{"id": "e1", "type": "embedding"}]},
            "tester",
        )


def test_normalize_payload_rejects_bad_provider_type():
    with pytest.raises(ValueError, match="provider_type"):
        svc._normalize_payload({"provider_id": "p1", "display_name": "P", "base_url": "https://x", "provider_type": "weird"})


def test_normalize_model_list_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="重复"):
        svc._normalize_model_list([{"id": "m1", "type": "chat"}, {"id": "m1", "type": "embedding"}])


# ============================================================ builtin seeding


async def test_ensure_builtin_seeds_templates(repos):
    await svc.ensure_builtin_model_providers_in_db()
    all_providers = await svc.get_all_model_providers()
    by_id = {p.provider_id: p for p in all_providers}
    assert len(by_id) == len(BUILTIN_PROVIDERS)

    silicon = by_id["siliconflow-cn"]
    assert silicon.is_enabled is True
    assert silicon.is_builtin is True
    assert silicon.api_key_env == "SILICONFLOW_API_KEY"
    embedding_ids = [m["id"] for m in silicon.enabled_models if m["type"] == "embedding"]
    assert "Qwen/Qwen3-Embedding-0.6B" in embedding_ids

    assert by_id["openai"].is_enabled is False
    assert by_id["openai"].is_builtin is True


async def test_ensure_builtin_is_idempotent_and_refills_empty_models(repos):
    await svc.ensure_builtin_model_providers_in_db()
    await svc.ensure_builtin_model_providers_in_db()
    assert len(await svc.get_all_model_providers()) == len(BUILTIN_PROVIDERS)

    # deleting enabled_models triggers the refill on the next ensure
    await svc.update_provider_config("siliconflow-cn", {"enabled_models": []}, "system")
    await svc.ensure_builtin_model_providers_in_db()
    refilled = await svc.get_model_provider_by_id("siliconflow-cn")
    assert refilled is not None
    template = next(p for p in BUILTIN_PROVIDERS if p["provider_id"] == "siliconflow-cn")
    assert len(refilled.enabled_models or []) == len(template["enabled_models"])


# ================================================================== cache


async def test_rebuild_skips_disabled_providers_and_filters_by_type():
    enabled = _provider(provider_id="p-enabled", enabled_models=[{"id": "c1", "type": "chat"}])
    disabled = _provider(provider_id="p-disabled", enabled_models=[{"id": "c2", "type": "chat"}], is_enabled=False)
    embedding = _provider(
        provider_id="p-embed",
        enabled_models=[{"id": "e1", "type": "embedding", "dimension": 16, "batch_size": 8}],
    )
    model_cache.rebuild([enabled, disabled, embedding])

    assert model_cache.get_model_info("p-disabled:c2") is None
    assert model_cache.get_model_info("p-enabled:c1") is not None

    embedding_specs = model_cache.get_all_specs("embedding")
    assert [info.spec for info in embedding_specs] == ["p-embed:e1"]
    assert embedding_specs[0].dimension == 16
    assert embedding_specs[0].batch_size == 8

    grouped = model_cache.get_specs_grouped_by_provider("chat")
    assert set(grouped) == {"p-enabled"}
    assert grouped["p-enabled"][0].spec == "p-enabled:c1"


def test_rebuild_picks_base_url_by_type_and_detects_vision():
    provider = _provider(
        provider_id="p-urls",
        base_url="https://a.test/v1",
        embedding_base_url="https://a.test/v1/embeddings",
        enabled_models=[
            {"id": "c1", "type": "chat"},
            {"id": "v1", "type": "chat", "extra": {"input_modalities": ["text", "image"]}},
        ],
    )
    model_cache.rebuild([provider])
    assert model_cache.get_model_info("p-urls:c1").base_url == "https://a.test/v1"
    assert model_cache.get_model_info("p-urls:c1").supports_vision is False
    assert model_cache.get_model_info("p-urls:v1").supports_vision is True


def test_rebuild_model_base_url_override():
    provider = _provider(
        provider_id="p-override",
        base_url="https://a.test/v1",
        enabled_models=[{"id": "c1", "type": "chat", "base_url_override": "https://custom.test/v1"}],
    )
    model_cache.rebuild([provider])
    assert model_cache.get_model_info("p-override:c1").base_url == "https://custom.test/v1"


def test_resolve_model_spec_hit_and_miss(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-1")
    model_cache.rebuild([_provider()])
    info = resolve_model_spec("test-provider:chat-model")
    assert info.model_type == "chat"
    assert info.spec == "test-provider:chat-model"

    with pytest.raises(ValueError, match="未找到模型"):
        resolve_model_spec("test-provider:nope")


def test_resolve_model_spec_empty_raises():
    with pytest.raises(ValueError, match="不能为空"):
        resolve_model_spec("")


# ============================================================ status probe


async def test_model_status_spec_not_found():
    result = await svc.test_model_status_by_spec("ghost:model")
    assert result["status"] == "error"
    assert "未找到模型" in result["message"]


async def test_model_status_embedding_available(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-1")
    model_cache.rebuild([_provider(enabled_models=[{"id": "e1", "type": "embedding", "dimension": 8}])])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 8}]})

    result = await svc.test_model_status_by_spec("test-provider:e1", transport=httpx.MockTransport(handler))
    assert result["status"] == "available"
    assert result["model_type"] == "embedding"


async def test_model_status_chat_available(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-1")
    model_cache.rebuild([_provider(enabled_models=[{"id": "c1", "type": "chat"}])])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": "1"}}]})

    result = await svc.test_model_status_by_spec("test-provider:c1", transport=httpx.MockTransport(handler))
    assert result["status"] == "available"
    assert result["model_type"] == "chat"


async def test_model_status_chat_unavailable(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-1")
    model_cache.rebuild([_provider(enabled_models=[{"id": "c1", "type": "chat"}])])

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    result = await svc.test_model_status_by_spec("test-provider:c1", transport=httpx.MockTransport(handler))
    assert result["status"] == "unavailable"


def test_model_info_dict_roundtrip():
    provider = _provider()
    model_cache.rebuild([provider])
    info = model_cache.get_model_info("test-provider:chat-model")
    assert info is not None
    restored = cache_module.ModelInfo.from_dict(info.to_dict())
    assert restored == info


# ============================================================ factory (two-level)


async def test_factory_chat_spec_and_env_fallback(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-1")
    model_cache.rebuild([_provider(enabled_models=[{"id": "c1", "type": "chat"}])])

    from yusu_kb.models.chat import create_chat_model

    adapter = create_chat_model(default_spec="test-provider:c1")
    assert adapter.model == "c1"
    assert adapter.base_url.endswith("/chat/completions")
    assert adapter.api_key == "sk-1"

    monkeypatch.setenv("YUSU_LLM_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("YUSU_LLM_MODEL", "env-model")
    monkeypatch.setenv("YUSU_LLM_API_KEY", "sk-env")
    adapter_env = create_chat_model()
    assert adapter_env.model == "env-model"
    assert adapter_env.base_url == "https://env.test/v1/chat/completions"

    with pytest.raises(ValueError, match="未找到模型"):
        create_chat_model(default_spec="test-provider:nope")


async def test_factory_chat_rejects_wrong_type(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-1")
    model_cache.rebuild([_provider(enabled_models=[{"id": "e1", "type": "embedding", "dimension": 8}])])

    from yusu_kb.models.chat import create_chat_model

    with pytest.raises(ValueError, match="不是 chat"):
        create_chat_model(default_spec="test-provider:e1")


async def test_factory_embedding_and_rerank_spec(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-1")
    model_cache.rebuild(
        [
            _provider(
                enabled_models=[
                    {"id": "e1", "type": "embedding", "dimension": 8},
                    {"id": "r1", "type": "rerank"},
                ]
            )
        ]
    )

    from yusu_kb.models.embed import create_embedding_model
    from yusu_kb.models.rerank import create_reranker

    embed = create_embedding_model(default_spec="test-provider:e1")
    assert embed.model == "e1"
    assert embed.dimension == 8
    assert embed.base_url == "https://provider.test/v1/embeddings"

    reranker = create_reranker(default_spec="test-provider:r1")
    assert reranker is not None
    assert reranker.model == "r1"

    with pytest.raises(ValueError, match="不是 rerank"):
        create_reranker(default_spec="test-provider:e1")
    with pytest.raises(ValueError, match="不是 embedding"):
        create_embedding_model(default_spec="test-provider:r1")


# ============================================================ API router


@pytest.fixture()
async def client(engine):
    from yusu_kb.api import deps as api_deps
    from yusu_kb.api.app import create_app

    configure_repositories(engine)
    transport_holder: dict = {}

    def _transport_override():
        return transport_holder.get("transport")

    app = create_app()
    app.dependency_overrides[api_deps.get_httpx_transport] = _transport_override
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        yield http_client, transport_holder
    app.dependency_overrides.clear()
    configure_repositories(None)


async def test_api_provider_crud_chain(client):
    http_client, _ = client
    response = await http_client.post(
        "/api/system/model-providers",
        json={
            "provider_id": "api-provider",
            "display_name": "API Provider",
            "base_url": "https://api-provider.test/v1",
            "api_key_env": "API_PROVIDER_KEY",
            "capabilities": ["chat"],
            "enabled_models": [{"id": "c1", "type": "chat"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["provider_id"] == "api-provider"
    assert body["data"]["credential_status"] == "warning"

    listing = await http_client.get("/api/system/model-providers")
    assert listing.status_code == 200
    assert any(p["provider_id"] == "api-provider" for p in listing.json()["data"])

    fetched = await http_client.get("/api/system/model-providers/api-provider")
    assert fetched.status_code == 200
    assert fetched.json()["data"]["display_name"] == "API Provider"

    updated = await http_client.put(
        "/api/system/model-providers/api-provider",
        json={"display_name": "改名", "is_enabled": False},
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["display_name"] == "改名"
    assert updated.json()["data"]["credential_status"] == "ok"

    assert (await http_client.put("/api/system/model-providers/missing", json={"display_name": "x"})).status_code == 404

    deleted = await http_client.delete("/api/system/model-providers/api-provider")
    assert deleted.status_code == 200
    assert (await http_client.delete("/api/system/model-providers/api-provider")).status_code == 404


async def test_api_provider_create_duplicate_400(client):
    http_client, _ = client
    payload = {
        "provider_id": "dup-provider",
        "display_name": "Dup",
        "base_url": "https://dup.test/v1",
    }
    assert (await http_client.post("/api/system/model-providers", json=payload)).status_code == 200
    duplicate = await http_client.post("/api/system/model-providers", json=payload)
    assert duplicate.status_code == 400
    assert "已存在" in duplicate.json()["detail"]


async def test_api_remote_models_and_401_to_502(client, monkeypatch):
    http_client, transport_holder = client
    monkeypatch.setenv("API_PROVIDER_KEY", "sk-ok")

    async def _seed_provider() -> str:
        response = await http_client.post(
            "/api/system/model-providers",
            json={
                "provider_id": "remote-provider",
                "display_name": "Remote",
                "base_url": "https://remote.test/v1",
                "api_key_env": "API_PROVIDER_KEY",
                "capabilities": ["chat"],
                "models_endpoint": "models",
            },
        )
        assert response.status_code == 200
        return "remote-provider"

    provider_id = await _seed_provider()

    def ok_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"data": [{"id": "r1", "name": "Remote One"}]})

    transport_holder["transport"] = httpx.MockTransport(ok_handler)
    fetched = await http_client.get(f"/api/system/model-providers/{provider_id}/remote-models")
    assert fetched.status_code == 200
    models = fetched.json()["data"]
    assert [m["id"] for m in models] == ["r1"]

    def auth_fail_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    transport_holder["transport"] = httpx.MockTransport(auth_fail_handler)
    failed = await http_client.get(f"/api/system/model-providers/{provider_id}/remote-models")
    assert failed.status_code == 502
    assert "认证失败" in failed.json()["detail"]

    missing = await http_client.get("/api/system/model-providers/nope/remote-models")
    assert missing.status_code == 404


async def test_api_cache_refresh_models_v2_and_status(client, monkeypatch):
    http_client, transport_holder = client
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-silicon")

    refreshed = await http_client.post("/api/system/model-providers/models/cache/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["success"] is True
    assert refreshed.json()["model_count"] > 0

    v2 = await http_client.get("/api/system/model-providers/models/v2", params={"model_type": "embedding"})
    assert v2.status_code == 200
    data = v2.json()["data"]
    assert "siliconflow-cn" in data
    embedding_specs = [m["spec"] for m in data["siliconflow-cn"]["models"]]
    assert "siliconflow-cn:Qwen/Qwen3-Embedding-0.6B" in embedding_specs

    def embed_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024}]})

    transport_holder["transport"] = httpx.MockTransport(embed_handler)
    status = await http_client.get(
        "/api/system/model-providers/models/status",
        params={"spec": "siliconflow-cn:Qwen/Qwen3-Embedding-0.6B"},
    )
    assert status.status_code == 200
    assert status.json()["data"]["status"] == "available"

    unknown = await http_client.get(
        "/api/system/model-providers/models/status",
        params={"spec": "ghost:model"},
    )
    assert unknown.status_code == 200
    assert unknown.json()["data"]["status"] == "error"
    assert "未找到模型" in unknown.json()["data"]["message"]


async def test_api_defaults_roundtrip(client):
    http_client, _ = client
    put = await http_client.put(
        "/api/system/model-providers/defaults",
        json={"default_chat_model_spec": "siliconflow-cn:deepseek-ai/DeepSeek-V4-Flash"},
    )
    assert put.status_code == 200
    assert put.json()["data"]["default_chat_model_spec"] == "siliconflow-cn:deepseek-ai/DeepSeek-V4-Flash"

    got = await http_client.get("/api/system/model-providers/defaults")
    assert got.status_code == 200
    data = got.json()["data"]
    assert data["default_chat_model_spec"] == "siliconflow-cn:deepseek-ai/DeepSeek-V4-Flash"
    assert data["default_embedding_model_spec"] is None
    assert data["default_rerank_model_spec"] is None
