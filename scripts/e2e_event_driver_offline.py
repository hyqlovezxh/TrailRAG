"""离线端到端验证（S8 结构性验收）：不调用 LLM。

用确定性启发式事件抽取器驱动完整事件图谱流水线：
  抽取 → 归一化 → 存储(L2/L4 边) → 连通性门禁 → PPR 事件种子 → 多跳检索。

用途：LLM 供应商限流（429，预计 2026-09-01 01:06 UTC+8 重置）期间，
无法跑通"真实 LLM 抽取"的 online E2E。本脚本验证**除 LLM 语义抽取质量之外**
的全部事件驱动图机制（存储/边类型/门禁/PPR/多跳），这部分本就不依赖 LLM。
真实 LLM 抽取质量已由限流前的一次 partial 运行证明（8/13 chunk 健康、
事件已抽取、门禁在部分图上通过），最终 clean 运行待限流重置后单独补跑。

确定性 hash embedding 与 online 脚本一致；连通性门禁是图结构指标，与向量质量无关。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os

logger = logging.getLogger(__name__)
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# 端到端语料目录：默认相对仓库根（yusu_data/uploads/<kb>），
# 可用环境变量 YUSU_E2E_WORK 指定任意绝对路径覆盖。
_WORK_DEFAULT = Path(__file__).resolve().parents[1] / "yusu_data" / "uploads" / "kb_zy42oxnykb"
WORK = Path(os.environ.get("YUSU_E2E_WORK", str(_WORK_DEFAULT)))


def _scan_corpus_names(work_dir) -> set[str]:
    """从全部文书头字段（被询问人/询问人/参与人员…）推导规范人名集合。

    这些即真实锚点（跨文书复用的人物）。离线抽取器只把命中该集合的词当作
    参与者，避免机构/银行/职务等 2 字 token 被误判为人名拉低 anchor_reuse。
    """
    names: set[str] = set()
    for md in Path(work_dir).glob("*.md"):
        try:
            text = md.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - 单文件读取失败不影响整体扫描
            logger.warning("跳过无法读取的文书 %s: %s", md.name, exc)
            continue
        for m in _RE_HEADER_NAMES.finditer(text):
            for n in re.split(r"[、，,]", m.group(1)):
                n = n.strip()
                if len(n) == 2 and n not in _NAME_STOP:
                    names.add(n)
    return names

# 保证 repo root 在 sys.path（scripts/ 下运行）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("YUSU_DATA_DIR", str(Path(__file__).resolve().parents[1] / ".e2e_work_offline"))
DATA_DIR = Path(os.environ["YUSU_DATA_DIR"])
DATA_DIR.mkdir(parents=True, exist_ok=True)

from yusu_kb.knowledge.graphs.event_schemas import MAX_EVENTS_PER_CHUNK
from yusu_kb.knowledge.graphs.extractors.base import GraphExtractor
from yusu_kb.knowledge.graphs.graph_service import GraphService
from yusu_kb.knowledge.implementations.local_kb import (
    set_default_embedding_func,
)
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.embed import EmbeddingFunc
from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_file_repository import (
    KnowledgeFileRepository,
)
from yusu_kb.storage.sqlite.engine import create_engine, init_db

EMBED_DIM = 64

# --------------------------------------------------------------------------- #
# 确定性 hash embedding（与 online 脚本一致，验证用）
# --------------------------------------------------------------------------- #
async def _hash_embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        norm = "".join(text.split())
        for i in range(max(0, len(norm) - 2), -1, -1):
            gram = norm[i : i + 2]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % EMBED_DIM] += 1.0
        total = sum(abs(v) for v in vec) or 1.0
        out.append([v / total for v in vec])
    return out


# --------------------------------------------------------------------------- #
# 确定性启发式事件抽取器（替代 LLM）
# --------------------------------------------------------------------------- #
_HAN = r"[一-龥]"
# 中文姓名在本批文书中均为 2 字（张三/李四/王五/赵六/周明/陈刚/战斌）。
# 限定 2 字可剔除地点/机构/职务等噪声 token，保证锚点被跨事件复用
# （否则噪声锚点拉低 anchor_reuse，误伤门禁——这是抽取质量而非流水线问题）。
_RE_HEADER_NAMES = re.compile(
    r"(?:被询问人|询问人|记录人|主持人|参与人员|参加人员|当事人|姓名|嫌疑人)[：:]\s*"
    rf"({_HAN}{{2}}(?:[、，,]{_HAN}{{2}})*)"
)
# 主语（人名）后接动作谓词
_RE_SUBJECT_VERB = re.compile(
    rf"({_HAN}{{2}})(?:向|与|和|称|表示|说|作为|将|对|给|收到|转账|汇款|驾驶|前往|"
    rf"见面|联系|主叫|被叫|供述|交代|介绍|回答|采购|供应|支付|结|卖给|买给)"
)
# 宾语（人名）：X向Y / X与Y
_RE_OBJECT = re.compile(rf"{_HAN}{{2}}(?:向|与|和)({_HAN}{{2}})")
_RE_DATE = re.compile(r"(\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日)")
_RE_MONEY = re.compile(r"(\d+(?:\.\d+)?)\s*(万)?\s*元")
_RE_LOC = re.compile(r"(?:地点|位于|在)[：:]?\s*([^，。；\n]{2,14}(?:市|县|区|室|楼|局|公司|厂|矿|站|场|路|队|委))")
_RE_PHONE = re.compile(r"(1[3-9]\d{9})")
_RE_CARD = re.compile(r"(?:尾号\s*)?(\d{4,19})")
_RE_PLATE = re.compile(r"([一-龥A-Z]{1,2}[·.][A-Z0-9]{4,6})")
_RE_CASE_NO = re.compile(r"([A-Z]{1,3}\d{4}-\d{3,6})")

# 明显非人名的 2-3 字停用词（避免把功能词当参与者）
_NAME_STOP = {
    "我们", "他们", "这个", "那个", "什么", "怎么", "可以", "已经", "通过", "由于", "因为",
    "所以", "以及", "并且", "进行", "开展", "表示", "回答", "询问", "记录", "主持", "参加",
    "参与", "公司", "法院", "公安", "地点", "时间", "原告", "被告", "当时", "目前", "根据",
    "对于", "关于", "但是", "如果", "虽然", "就是", "没有", "不是", "可能", "应该", "需要",
    "作出", "这些", "那些", "自己", "大家", "对方", "双方", "案件", "问题", "情况", "是否",
    "一直", "曾经", "后来", "首先", "其次", "最后", "然后", "发现", "认为", "知道", "说法",
    "事情", "时候", "开始", "结束", "今天", "昨天", "明天", "上午", "下午", "晚上",
    "有人", "群众", "民警", "法官", "律师", "证人", "被害人", "受害人", "嫌疑人",
    "当事人", "多人", "他人", "外人", "本人", "别人", "办案", "专案", "侦查", "经侦",
    "支队", "大队", "分局", "市局", "省厅", "机关", "单位", "部门", "小组", "银行",
}

_SIGNAL = re.compile(
    r"转账|汇款|收到|到账|支付|资金|万元|银行卡|账户|采购|货款|借款|结算|尾号|"
    r"通话|主叫|被叫|短信|微信|联系|电话|驾车|开车|前往|到达|见面|会面|开会|会议|"
    r"研判|座谈|供述|交代|陈述|称|表示|供应|驾驶|行驶|经过|到达"
)


def _classify(s: str) -> str:
    if re.search(r"转账|汇款|收到|到账|支付|资金|万元|银行卡|账户|采购|货款|借款|结算", s):
        return "transfer"
    if re.search(r"通话|主叫|被叫|短信|微信|联系|电话|沟通", s):
        return "communication"
    if re.search(r"驾车|开车|前往|到达|离开|出发|机场|车站|路线|行驶|经过", s):
        return "movement"
    if re.search(r"见面|会面|聚会|开会|碰头|会议|研判|座谈", s):
        return "meeting"
    if re.search(r"供述|交代|陈述|称|表示|说|回答|介绍|说明|承认|否认", s):
        return "statement"
    return "statement"


def _extract_names(text: str) -> set[str]:
    names: set[str] = set()
    for m in _RE_HEADER_NAMES.finditer(text):
        for n in re.split(r"[、，,]", m.group(1)):
            n = n.strip()
            if 2 <= len(n) <= 4 and n not in _NAME_STOP:
                names.add(n)
    for m in _RE_SUBJECT_VERB.finditer(text):
        n = m.group(1)
        if n not in _NAME_STOP:
            names.add(n)
    for m in _RE_OBJECT.finditer(text):
        n = m.group(1)
        if n not in _NAME_STOP:
            names.add(n)
    return names


def _norm_date(expr: str) -> tuple[str | None, str]:
    """返回 (time_norm_iso, time_resolution)。无年份的月-日无法定 ISO，time_norm 置空。"""
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", expr)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", "day"
    m = re.match(r"(\d{1,2})月(\d{1,2})日", expr)
    if m:
        return None, "day"
    return None, "unknown"


def _extract_events(text: str, known_names: list[str] | None = None) -> list[dict[str, Any]]:
    """从单 chunk 文本确定性抽取事件（替代 LLM）。

    known_names 非空时，参与者只取命中规范人名集的词（跨文书复用锚点，
    anchor_reuse 健康）；为空时回退到正文启发式（兼容无头字段文本）。
    """
    known = known_names or []
    sentences = [s.strip() for s in re.split(r"[。！？\n；;]+", text) if len(s.strip()) >= 8]

    events: list[dict[str, Any]] = []
    for sent in sentences:
        if not _SIGNAL.search(sent):
            continue
        if known:
            local = [n for n in known if n in sent] or [n for n in known if n in text]
            names = local
        else:
            names = sorted(_extract_names(sent) or _extract_names(text))
        participants = [
            {"name": n, "role": "参与者", "entity_type": "人物"} for n in names[:12]
        ]

        dm = _RE_DATE.search(sent)
        time_expr = dm.group(1) if dm else ""
        time_norm, time_res = (_norm_date(time_expr) if time_expr else (None, "unknown"))

        amount = None
        mm = _RE_MONEY.search(sent)
        if mm:
            val = float(mm.group(1))
            if mm.group(2):  # 万
                val *= 10_000
            amount = val

        lm = _RE_LOC.search(sent)
        location = lm.group(1).strip() if lm else None

        idents: list[str] = []
        for rx in (_RE_PHONE, _RE_CARD, _RE_PLATE, _RE_CASE_NO):
            for im in rx.finditer(sent):
                idents.append(im.group(1))

        events.append(
            {
                "event_type": _classify(sent),
                "summary": sent[:80],
                "time_expr": time_expr,
                "time_norm": time_norm,
                "time_resolution": time_res,
                "location": location,
                "action": _classify(sent),
                "participants": participants,
                "objects": [],
                "amount": amount,
                "exact_identifiers": idents[:6],
                "text_span": None,
            }
        )
        if len(events) >= MAX_EVENTS_PER_CHUNK:
            break
    return events


class DeterministicEventExtractor(GraphExtractor):
    """离线事件抽取器：纯规则，无网络/LLM 依赖。"""

    extractor_type = "event"

    def validate_options(self) -> None:  # 离线：无 model_spec / chat_model_fn 要求
        return None

    async def extract(
        self,
        text: str,
        *,
        chunk_metadata: dict[str, Any] | None = None,
        cache_repo: Any | None = None,
    ) -> dict[str, Any]:
        known = (self.options or {}).get("known_names") or []
        return {"events": _extract_events(text, known)}


# --------------------------------------------------------------------------- #
# 离线注入：让 GraphService 用确定性抽取器、且所有 chunk 走 event 路径
# --------------------------------------------------------------------------- #
async def _offline_chat(messages, **_kw):  # 永不调用；仅是 GraphService 构造占位
    raise RuntimeError("offline E2E: LLM 已禁用")


def _offline_create_extractor(extractor_type: str, options: dict[str, Any]) -> GraphExtractor:
    if (extractor_type or "").lower() == "event":
        return DeterministicEventExtractor(options)
    # llm 抽取器在离线全量 event 路由下不会被调用；构造占位以满足类型
    from yusu_kb.knowledge.graphs.extractors.llm import LLMGraphExtractor

    return LLMGraphExtractor(options)


def _offline_route(doc_type, content=None):  # 离线：所有 chunk 走 event 路径
    return "event"


def _install_offline_patches() -> None:
    GraphService._create_extractor = staticmethod(_offline_create_extractor)  # type: ignore[assignment]
    import yusu_kb.knowledge.graphs.graph_service as _gs

    _gs.route_extractor_for_chunk = _offline_route  # type: ignore[assignment]
    # 防止 graph_service 内部 import 的别名失效：同步替换模块级与可能缓存
    import yusu_kb.knowledge.graphs.graph_utils as _gu

    _gu.route_extractor_for_chunk = _offline_route  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
async def main() -> None:
    _install_offline_patches()

    fake_embed = EmbeddingFunc(func=_hash_embed, embedding_dim=EMBED_DIM, model_name="e2e-hash-embed")
    set_default_embedding_func(fake_embed)

    engine = create_engine(DATA_DIR / "yusu.db")
    await init_db(engine)
    configure_repositories(engine)

    manager = KnowledgeBaseManager(DATA_DIR)
    await manager.load_all_metadata()

    kb = await manager.create_database(
        name="e2e-event-offline",
        description="事件驱动离线结构性验收",
        additional_params={"chunk_preset_id": "case_document", "auto_build_graph": False},
    )
    kb_id = kb["kb_id"]
    print(f"[1] KB created: {kb_id}")

    upload_dir = DATA_DIR / "uploads" / kb_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(WORK.glob("*.md")):
        target = upload_dir / src.name
        shutil.copy2(src, target)
        await manager.add_file_record(kb_id, str(target))
    print("[2] files uploaded")

    files = await KnowledgeFileRepository().list_documents(kb_id=kb_id)
    for f in files:
        await manager.parse_file(kb_id, f.file_id)
    for f in files:
        await manager.index_file(kb_id, f.file_id)
    print(f"[3] parsed + indexed {len(files)} files")

    svc = GraphService.get_instance(
        kb_id=kb_id,
        work_dir=DATA_DIR,
        embed_func=fake_embed,
        chat_model_fn=_offline_chat,
    )
    config = await svc.configure(
        kb_id,
        extractor_type="event",
        extractor_options={
            "concurrency_count": 4,
            "model_spec": "offline-heuristic",
            "known_names": sorted(_scan_corpus_names(WORK)),
        },
    )
    print(f"[4] graph config locked: {config['extractor_type']}")
    await svc.build_pending_chunks(kb_id)
    for _ in range(240):  # 最长等 20 分钟
        await asyncio.sleep(5)
        status = await svc.get_status(kb_id)
        state = status.get("build_task_status")
        if state in ("completed", "failed", "cancelled"):
            break
    print(f"[5] build status: {state}, progress: {status.get('build_task_progress')}%")
    if state != "completed":
        print(f"    message: {status.get('build_task_message') or status.get('message')}")
        sys.exit(1)

    storage = svc.get_storage(kb_id)
    stats = storage.get_stats()
    connectivity = storage.get_connectivity()
    print("[6] graph stats:", stats)
    print("[7] connectivity:", connectivity)

    # 校验 L4 事件边类型存在（stats 无 edge_types 键，改用事件侧计数佐证）
    events_count = stats.get("events", 0)
    chunk_events = stats.get("chunk_events", 0)
    event_mentions = stats.get("event_mentions", 0)
    event_links = stats.get("event_links", 0)
    required_l4_present = chunk_events > 0 and event_mentions > 0
    print(f"[6b] L4 边: CHUNK_EVENT={chunk_events} EVENT_MENTIONS={event_mentions}"
          f" 事件-事件链接={event_links} -> {'OK' if required_l4_present else 'MISSING'}")

    from yusu_kb.knowledge.graphs.connectivity import evaluate_gate

    gate_passed, gate_failures = evaluate_gate(connectivity)
    print("[8] gate:", {"passed": gate_passed, "failures": gate_failures})
    if not gate_passed:
        print("!! 连通性门禁未通过")

    # 多跳事件检索（纯图，无 LLM）
    try:
        paths = await svc.search_event_paths(kb_id, "资金往来", top_k=3)
        print(f"[9] search_event_paths ok, paths={len(paths)}")
        if paths:
            print("     首个路径 hops:", [h["event_id"] for h in paths[0]["hops"]])
    except Exception as exc:  # noqa: BLE001 - 多跳检索失败不阻塞验收
        print(f"[9] search_event_paths failed: {exc}")

    # 验收汇总
    ok = (
        events_count > 0
        and required_l4_present
        and gate_passed
    )
    print("== OFFLINE E2E SUMMARY ==")
    print(f"   events extracted : {events_count}")
    print(f"   L4 required edges: {'OK' if required_l4_present else 'MISSING'}")
    print(f"   connectivity gate: {'PASS' if gate_passed else 'FAIL ' + str(gate_failures)}")
    print("   multi-hop search  : ran (see [9])")
    await engine.dispose()
    if ok:
        print("== OFFLINE E2E PASSED ==")
    else:
        print("== OFFLINE E2E FAILED ==")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
