"""Implicit-graph parameter calibration benchmark (plan Phase 4).

Grid-searches ``sim_threshold x knn_k x damping`` on a deterministic synthetic
multi-hop corpus (no LLM, no network, no external data), reports the metrics
from the plan's acceptance table and calibrates the ``max_corpus_nodes``
threshold against the actual machine BLAS.

Corpus design mirrors the production story: an identifier (bank card number)
hard-links chunks across topic clusters (EXACT_ID edges), narrative answers
sit next to the linked chunk (ADJACENT edges), and the query vector only
reaches the seed cluster -- so pure vector retrieval cannot reach the gold
chunks, but PPR over the implicit graph can.

Usage:
    ./.venv/Scripts/python.exe scripts/bench_implicit_graph.py [--quick] [--json out.json]

Acceptance gates (evaluated against the pure-vector baseline):
    MultiHop-Hit@10 >= +12pp, Hubness Index < 0.05, p95 latency <= +60ms.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make the repository package importable when the script is run directly
# (scripts/ sits one level below the repo root where the ```yusu_kb``` package lives).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from yusu_kb.knowledge.eval.metrics import RetrievalMetrics
from yusu_kb.knowledge.graphs.implicit_graph import (
    ImplicitChunkGraph,
    ImplicitGraphConfig,
)
from yusu_kb.knowledge.retrieval.query_analysis import analyze_query
from yusu_kb.storage.vector_store import VectorSnapshot

DIM = 64
SEED = 42
TOPICS = 12
DECOYS_PER_TOPIC = 20
CASES = 20
SEEDS_PER_CASE = 3
QUERIES_PER_CASE = 2
TOP_K = 10

# Acceptance gates from the plan (deltas vs the pure-vector baseline).
GATE_MULTIHOP_DELTA_PP = 12.0
GATE_HUBNESS = 0.05
GATE_P95_DELTA_MS = 60.0
BUILD_TIME_BUDGET_MS = 400.0


@dataclass
class BenchQuery:
    query_id: str
    text: str
    vector: np.ndarray
    gold: list[str]
    vector_top: list[str] = field(default_factory=list)


def _card_number(case: int) -> str:
    return f"622202{case:016d}"


def build_corpus() -> tuple[VectorSnapshot, list[BenchQuery]]:
    """Deterministic synthetic corpus + multi-hop queries.

    Background: ``TOPICS`` dense topic clusters of plain decoy chunks (no
    identifier). The vector channel retrieves these for any query, so pure
    vector ranking can never reach the answer.

    Each case plants a genuine 2-hop path that only the implicit graph can
    traverse:
      seed  (near topic centre, contains card C)            <- found by vector
        --EXACT_ID(card C)--> gold   (isolated vector, contains card C)
          --ADJACENT(chunk +1)--> gold_adj (isolated vector, contains card C)
    Gold chunks sit on isolated random vectors, so they are invisible to the
    vector channel yet reachable through the identifier's hard link.
    """
    rng = np.random.default_rng(SEED)
    centers = rng.normal(size=(TOPICS, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    contents: list[str] = []
    vectors: list[np.ndarray] = []
    file_ids: list[str] = []
    chunk_indexes: list[int] = []
    chunk_ids: list[str] = []
    case_gold: dict[int, list[str]] = {case: [] for case in range(CASES)}

    row = 0

    # 1) Background decoys: dense topic clusters without any identifier.
    for topic in range(TOPICS):
        file_id = f"topic{topic:02d}"
        for position in range(DECOYS_PER_TOPIC):
            vec = centers[topic] + rng.normal(scale=0.05, size=DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            chunk_id = f"ck{row:04d}"
            contents.append(f"主题{topic}常规台账第{position}段，日常业务流水记录。")
            vectors.append(vec)
            file_ids.append(file_id)
            chunk_indexes.append(position)
            chunk_ids.append(chunk_id)
            row += 1

    # 2) Per-case evidence. The seed cluster is a TIGHT isolated blob (its only
    # vector neighbours are the other seeds); the gold pair is FULLY isolated
    # (no SIM/ADJACENT edges to anything but each other). The card number is the
    # sole bridge: seeds --EXACT_ID(card)--> gold --ADJACENT--> gold_adj. Because
    # the seed cluster has no other outgoing edges, PPR mass must flow onto gold.
    for case in range(CASES):
        card = _card_number(case)
        file_id = f"case{case:02d}"
        u = rng.normal(size=DIM).astype(np.float32)
        u /= np.linalg.norm(u)
        for s in range(SEEDS_PER_CASE):
            seed_vec = u + rng.normal(scale=0.04, size=DIM).astype(np.float32)
            seed_vec /= np.linalg.norm(seed_vec)
            seed_id = f"ck{row:04d}"
            contents.append(f"线索{s}：账户{card}涉案资金首次出现的流水。")
            vectors.append(seed_vec)
            file_ids.append(file_id)
            chunk_indexes.append(s)
            chunk_ids.append(seed_id)
            row += 1
        for j, cidx in enumerate((5, 6), start=1):
            gv = rng.normal(size=DIM).astype(np.float32)
            gv /= np.linalg.norm(gv)
            gid = f"ck{row:04d}"
            contents.append(
                f"账户{card}资金最终汇入对象及关联账户说明（证据{j}）。"
            )
            vectors.append(gv)
            file_ids.append(file_id)
            chunk_indexes.append(cidx)
            chunk_ids.append(gid)
            case_gold[case].append(gid)
            row += 1

    matrix = np.asarray(vectors, dtype=np.float32)
    metas = [
        {"content": content, "file_id": file_id, "chunk_index": index}
        for content, file_id, index in zip(contents, file_ids, chunk_indexes)
    ]
    snapshot = VectorSnapshot(ids=chunk_ids, matrix=matrix, metas=metas)

    queries: list[BenchQuery] = []
    for case in range(CASES):
        card = _card_number(case)
        # Reconstruct the seed-cluster direction for this case: the first seed
        # chunk sits at index (decoy_count + case * (SEEDS_PER_CASE + 2)).
        seed_base = TOPICS * DECOYS_PER_TOPIC + case * (SEEDS_PER_CASE + 2)
        u = vectors[seed_base] / np.linalg.norm(vectors[seed_base])
        for q in range(QUERIES_PER_CASE):
            vec = u + rng.normal(scale=0.02, size=DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            sims = matrix @ vec
            vector_top = [chunk_ids[i] for i in np.argsort(-sims)[:TOP_K]]
            queries.append(
                BenchQuery(
                    query_id=f"case{case}-q{q}",
                    text=f"账户{card}的资金流向",
                    vector=vec,
                    gold=sorted(case_gold[case]),
                    vector_top=vector_top,
                )
            )
    return snapshot, queries


def _seed_weights(snapshot: VectorSnapshot, query: BenchQuery, top: int = 5) -> dict[str, float]:
    sims = snapshot.matrix @ query.vector
    order = np.argsort(-sims)[:top]
    raw = {snapshot.ids[int(i)]: float(sims[int(i)]) for i in order if sims[int(i)] > 0.0}
    total = sum(raw.values())
    if total <= 0.0:
        return {snapshot.ids[int(order[0])]: 1.0}
    return {chunk_id: weight / total for chunk_id, weight in raw.items()}


def evaluate_config(
    snapshot: VectorSnapshot, queries: list[BenchQuery], cfg: ImplicitGraphConfig
) -> dict:
    """Build the corpus graph once, then rank every query and score it."""
    build_started = time.perf_counter()
    graph = ImplicitChunkGraph.build_full(kb_id="bench", snapshot=snapshot, cfg=cfg)
    build_ms = (time.perf_counter() - build_started) * 1000.0

    query_latencies: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    multihop_hits: list[float] = []
    new_chunk_rates: list[float] = []

    for query in queries:
        started = time.perf_counter()
        analysis = analyze_query(query.text)
        seeds = _seed_weights(snapshot, query)
        query_cosine = graph.query_cosine(query.vector)
        nx_graph = graph.to_networkx(
            seed_ids=list(seeds),
            query_terms=analysis.scoring_terms,
            exact_tokens=analysis.exact_tokens,
        )
        ranked = graph.rank(nx_graph, query_cosine=query_cosine, seed_weights=seeds, top_k=TOP_K)
        query_latencies.append((time.perf_counter() - started) * 1000.0)

        top_ids = [chunk_id for chunk_id, _ in ranked]
        gold_set = query.gold
        recall_scores.append(RetrievalMetrics.recall_at_k(top_ids, gold_set, TOP_K))
        ndcg_scores.append(RetrievalMetrics.ndcg_at_k(top_ids, gold_set, TOP_K))
        multihop_hits.append(1.0 if all(chunk_id in top_ids for chunk_id in gold_set) else 0.0)
        vector_top10 = set(query.vector_top[:TOP_K])
        new_chunk_rates.append(
            sum(1 for chunk_id in top_ids if chunk_id not in vector_top10) / max(len(top_ids), 1)
        )

    latencies = sorted(query_latencies)
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    return {
        "sim_threshold": cfg.sim_threshold,
        "knn_k": cfg.knn_k,
        "damping": cfg.damping,
        "recall_at_10": float(np.mean(recall_scores)),
        "ndcg_at_10": float(np.mean(ndcg_scores)),
        "multihop_hit_at_10": float(np.mean(multihop_hits)),
        "new_chunk_rate": float(np.mean(new_chunk_rates)),
        "hub_index": float(graph.hub_index),
        "build_ms": build_ms,
        "p95_ms": p95,
        "tier": graph.tier,
        "edges": graph.static_edge_count,
    }


def evaluate_baseline(queries: list[BenchQuery], gold_weighted: bool = False) -> dict:
    """Pure-vector ranking (the degradation path the fallback replaces)."""
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    multihop_hits: list[float] = []
    for query in queries:
        top_ids = query.vector_top[:TOP_K]
        gold_set = query.gold
        recall_scores.append(RetrievalMetrics.recall_at_k(top_ids, gold_set, TOP_K))
        ndcg_scores.append(RetrievalMetrics.ndcg_at_k(top_ids, gold_set, TOP_K))
        multihop_hits.append(1.0 if all(chunk_id in top_ids for chunk_id in gold_set) else 0.0)
    return {
        "recall_at_10": float(np.mean(recall_scores)),
        "ndcg_at_10": float(np.mean(ndcg_scores)),
        "multihop_hit_at_10": float(np.mean(multihop_hits)),
    }


def calibrate_max_corpus_nodes(cfg: ImplicitGraphConfig) -> dict:
    """Measure full-build time against N and find the largest N within budget."""
    rng = np.random.default_rng(SEED + 1)
    results: list[dict] = []
    for n in (500, 1000, 2000, 3000, 4000, 5000, 6000):
        matrix = rng.normal(size=(n, DIM)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        snapshot = VectorSnapshot(
            ids=[f"cal{i:05d}" for i in range(n)],
            matrix=matrix,
            metas=[{"content": f"标定分块{i}", "file_id": "f", "chunk_index": i} for i in range(n)],
        )
        started = time.perf_counter()
        graph = ImplicitChunkGraph.build_full(kb_id="calibration", snapshot=snapshot, cfg=cfg)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results.append({"n": n, "build_ms": round(elapsed_ms, 1), "hub_index": round(float(graph.hub_index), 4)})
    within_budget = [r["n"] for r in results if r["build_ms"] <= BUILD_TIME_BUDGET_MS]
    return {
        "measurements": results,
        "budget_ms": BUILD_TIME_BUDGET_MS,
        "recommended_max_corpus_nodes": max(within_budget) if within_budget else None,
    }


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="shrink the grid and query set")
    parser.add_argument("--json", dest="json_path", default=None, help="also write results as JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    snapshot, queries = build_corpus()
    if args.quick:
        queries = queries[:10]
    print(f"corpus: {len(snapshot.ids)} chunks / {TOPICS} topics; queries: {len(queries)}")
    print()

    baseline = evaluate_baseline(queries)
    print("baseline (pure vector top-k):")
    print(
        f"  recall@10={baseline['recall_at_10']:.3f}  ndcg@10={baseline['ndcg_at_10']:.3f}  "
        f"multihop@10={baseline['multihop_hit_at_10']:.3f}"
    )
    print()

    grid = [
        (sim, knn, damping)
        for sim in (0.45, 0.55, 0.65)
        for knn in (8, 12, 20)
        for damping in (0.75, 0.80, 0.85)
    ]
    if args.quick:
        grid = grid[::4]

    rows: list[dict] = []
    print(f"grid search over {len(grid)} configs (build + {len(queries)} queries each):")
    header = (
        f"{'sim':>5} {'knn':>4} {'damp':>5} | {'R@10':>6} {'nDCG@10':>8} {'MH@10':>6} "
        f"{'NewChunk':>8} {'Hub':>6} | {'build_ms':>8} {'p95_ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for sim, knn, damping in grid:
        cfg = ImplicitGraphConfig(
            sim_threshold=sim, knn_k=knn, damping=damping, max_corpus_nodes=len(snapshot.ids)
        )
        row = evaluate_config(snapshot, queries, cfg)
        rows.append(row)
        print(
            f"{sim:>5.2f} {knn:>4d} {damping:>5.2f} | "
            f"{row['recall_at_10']:>6.3f} {row['ndcg_at_10']:>8.3f} {row['multihop_hit_at_10']:>6.3f} "
            f"{row['new_chunk_rate']:>8.3f} {row['hub_index']:>6.3f} | "
            f"{row['build_ms']:>8.1f} {row['p95_ms']:>7.1f}"
        )
    print()

    best = max(rows, key=lambda r: (r["multihop_hit_at_10"], r["ndcg_at_10"], -r["p95_ms"]))
    multihop_delta_pp = (best["multihop_hit_at_10"] - baseline["multihop_hit_at_10"]) * 100.0
    p95_delta_ms = best["p95_ms"]
    gates = {
        "multihop_delta_pp": (multihop_delta_pp, GATE_MULTIHOP_DELTA_PP, ">="),
        "hubness": (best["hub_index"], GATE_HUBNESS, "<"),
        "p95_delta_ms": (p95_delta_ms, GATE_P95_DELTA_MS, "<="),
    }
    print("best config by MultiHop-Hit@10:")
    print(
        f"  sim_threshold={best['sim_threshold']} knn_k={best['knn_k']} damping={best['damping']} "
        f"tier={best['tier']} edges={best['edges']}"
    )
    print("acceptance gates:")
    all_passed = True
    for name, (value, target, op) in gates.items():
        passed = value >= target if op == ">=" else value <= target
        all_passed = all_passed and passed
        print(f"  {name}: {value:.3f} {op} {target:.3f} -> {'PASS' if passed else 'FAIL'}")
    print()

    print("max_corpus_nodes calibration (full-build wall time on this machine):")
    calibration = calibrate_max_corpus_nodes(
        ImplicitGraphConfig(sim_threshold=best["sim_threshold"], knn_k=best["knn_k"])
    )
    for m in calibration["measurements"]:
        print(f"  N={m['n']:>5d}  build={m['build_ms']:>8.1f}ms  hub={m['hub_index']:.4f}")
    print(f"  recommended max_corpus_nodes: {calibration['recommended_max_corpus_nodes']}")

    if args.json_path:
        payload = {
            "baseline": baseline,
            "grid": rows,
            "best": best,
            "gates": {k: {"value": v[0], "target": v[1], "op": v[2]} for k, v in gates.items()},
            "calibration": calibration,
            "all_gates_passed": all_passed,
        }
        await asyncio.to_thread(_write_json, args.json_path, payload)
        print(f"\nresults written to {args.json_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
