from __future__ import annotations
import os
import json
import random
import time
import argparse
import re
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from ehr_rlgraph.memory_graph import GraphMemory
from ehr_rlgraph.rl_retriever import RLRetriever, RLRetrieverConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# def _get_question(x: Dict[str, Any]) -> str:
#     return (x.get("template") or x.get("question") or x.get("query") or "").strip()

def _get_question(x: Dict[str, Any]) -> str:
    return (
        x.get("template")
        or x.get("question")
        or x.get("question_refine")
        or x.get("query")
        or x.get("sql")
        or ""
    ).strip()


# def _get_id(x: Dict[str, Any], fallback: int) -> str:
#     v = x.get("id", None)
#     if v is None:
#         v = fallback
#     return str(v)

def _get_id(x: Dict[str, Any], fallback: int) -> str:
    v = x.get("id", None)
    if v is None:
        v = x.get("key", None)
    if v is None:
        v = fallback
    return str(v)

def _safe_meta_id(node: Any) -> Optional[str]:
    meta = getattr(node, "meta", None)
    if isinstance(meta, dict) and meta.get("id", None) is not None:
        return str(meta["id"])
    return None


# def build_gt_index(gm: GraphMemory) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
#     id2nodes: Dict[str, List[int]] = {}
#     q2nodes: Dict[str, List[int]] = {}

#     nodes = getattr(gm, "nodes", []) or []
#     for i, n in enumerate(nodes):
#         q = str(getattr(n, "q", "") or "").strip()
#         if q:
#             q2nodes.setdefault(q, []).append(i)

#         mid = _safe_meta_id(n)
#         if mid is not None:
#             id2nodes.setdefault(mid, []).append(i)

#     return id2nodes, q2nodes

def build_gt_index(gm: GraphMemory) -> Dict[str, List[int]]:
    id2nodes: Dict[str, List[int]] = {}
    nodes = getattr(gm, "nodes", []) or []

    for i, n in enumerate(nodes):
        mid = _safe_meta_id(n)
        if mid is not None:
            id2nodes.setdefault(mid, []).append(i)

    return id2nodes


def hit_at_k(selected: List[int], gt_nodes: List[int]) -> int:
    if not selected or not gt_nodes:
        return 0
    s = set(selected)
    return 1 if any((g in s) for g in gt_nodes) else 0

# Some helper functions for the RL update
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> Set[str]:
    text = _normalize_text(text)
    if not text:
        return set()
    return set(text.split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return _safe_div(len(a & b), len(a | b))


def _infer_question_type(text: str) -> str:
    t = _normalize_text(text)

    if any(x in t for x in ["how many", "number of", "count ", "total number", "total patients"]):
        return "count"
    if any(x in t for x in ["average", "mean"]):
        return "mean"
    if "maximum" in t or "max " in t or "highest" in t:
        return "max"
    if "minimum" in t or "min " in t or "lowest" in t:
        return "min"
    if any(x in t for x in ["list", "which patients", "what are", "show me all"]):
        return "list"
    return "lookup"


def _infer_answer_style(text: str) -> str:
    t = _normalize_text(text)

    if any(x in t for x in ["how many", "number of", "count ", "average", "mean", "minimum", "maximum", "min ", "max ", "total number"]):
        return "scalar"

    if any(x in t for x in ["list", "which patients", "what are", "show me all"]):
        return "list"

    return "lookup"


def _node_struct_profile(node: Any) -> Dict[str, Any]:
    q = str(getattr(node, "q", "") or "")
    p = str(getattr(node, "p", "") or "")
    s = getattr(node, "s", {}) or {}

    tables = set(s.get("tables", []) or []) if isinstance(s, dict) else set()
    columns = set(s.get("columns", []) or []) if isinstance(s, dict) else set()

    return {
        "q_tokens": _tokenize(q),
        "plan_tokens": _tokenize(p),
        "tables": set(str(x) for x in tables),
        "columns": set(str(x) for x in columns),
        "qtype": _infer_question_type(q),
        "ans_style": _infer_answer_style(q),
    }


def _avg_best_overlap(selected_profiles: List[Dict[str, Any]], gold_profiles: List[Dict[str, Any]], key: str) -> float:
    if not selected_profiles or not gold_profiles:
        return 0.0

    vals = []
    for sp in selected_profiles:
        best = 0.0
        for gp in gold_profiles:
            a = sp.get(key, set())
            b = gp.get(key, set())
            if isinstance(a, set) and isinstance(b, set):
                best = max(best, _jaccard(a, b))
        vals.append(best)

    return float(np.mean(vals)) if vals else 0.0


def _avg_best_exact_match(selected_profiles: List[Dict[str, Any]], gold_profiles: List[Dict[str, Any]], key: str) -> float:
    if not selected_profiles or not gold_profiles:
        return 0.0

    vals = []
    for sp in selected_profiles:
        best = 0.0
        for gp in gold_profiles:
            best = max(best, 1.0 if sp.get(key) == gp.get(key) else 0.0)
        vals.append(best)

    return float(np.mean(vals)) if vals else 0.0


def _compute_shaped_reward(
    gm: GraphMemory,
    selected: List[int],
    gt_nodes: List[int],
    reward_neg: float = 0.0,
) -> float:
    if not selected or not gt_nodes:
        return float(reward_neg)

    selected_set = set(selected)
    gt_set = set(gt_nodes)

    exact_hit = 1.0 if len(selected_set & gt_set) > 0 else 0.0

    selected_profiles = [_node_struct_profile(gm.get_node(i)) for i in selected]
    gold_profiles = [_node_struct_profile(gm.get_node(i)) for i in gt_nodes]

    token_overlap = _avg_best_overlap(selected_profiles, gold_profiles, "q_tokens")
    plan_overlap = _avg_best_overlap(selected_profiles, gold_profiles, "plan_tokens")
    table_overlap = _avg_best_overlap(selected_profiles, gold_profiles, "tables")
    column_overlap = _avg_best_overlap(selected_profiles, gold_profiles, "columns")

    qtype_match = _avg_best_exact_match(selected_profiles, gold_profiles, "qtype")
    ans_style_match = _avg_best_exact_match(selected_profiles, gold_profiles, "ans_style")

    # weighted shaped reward
    reward = (
        0.60 * exact_hit
        + 0.10 * token_overlap
        + 0.10 * table_overlap
        + 0.08 * column_overlap
        + 0.05 * plan_overlap
        + 0.04 * qtype_match
        + 0.03 * ans_style_match
    )

    if exact_hit == 0.0:
        reward += float(reward_neg)

    return float(reward)


def save_policy(retriever: RLRetriever, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    #get_state()
    # if hasattr(retriever, "save") and callable(getattr(retriever, "save")):
    #     retriever.save(path)
    #     # print(f"[INFO] Saved retriever state via save() -> {path}")
    #     return

    # for attr in ("theta", "weights", "w"):
    #     if hasattr(retriever, attr):
    #         obj = getattr(retriever, attr)
    #         try:
    #             payload = {"param_name": attr, "param": obj}
    #             with open(path, "w", encoding="utf-8") as f:
    #                 json.dump(payload, f, indent=2)
    #             print(f"[INFO] Saved retriever param '{attr}' -> {path}")
    #             return
    #         except Exception:
    #             pass
    if hasattr(retriever, "get_state") and callable(getattr(retriever, "get_state")):
        state = retriever.get_state()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"[INFO] Saved retriever state via get_state() -> {path}")
        return

    print("[WARN] Could not save policy. Add get_state() to RLRetriever.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mimic_iii", choices=["mimic_iii", "eicu", "treqs"])

    # Two possible training sources:
    parser.add_argument("--data_path", type=str, default="", help="Full dataset JSON (old mode)")
    parser.add_argument("--dgt_jsonl", type=str, default="", help="D_GT jsonl file (recommended mode)")

    parser.add_argument("--graph_path", type=str, required=True)

    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_policy", type=str, default="")
    parser.add_argument("--save_every", type=int, default=200)

    parser.add_argument("--seed_top_m", type=int, default=20)
    parser.add_argument("--expand_hops", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--entropy_bonus", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward_neg", type=float, default=0.0)

    args = parser.parse_args()
    set_seed(args.seed)

    # Load training questions
    # items = _read_json(args.data_path)
    # if not items:
    #     raise RuntimeError(f"Empty data_path: {args.data_path}")

    if args.dgt_jsonl:
        items = _read_jsonl(args.dgt_jsonl)
        print(f"[INFO] Training directly on D_GT file: {args.dgt_jsonl}")
    elif args.data_path:
        items = _read_json(args.data_path)
        print(f"[INFO] Training on dataset file: {args.data_path}")
    else:
        raise RuntimeError("Must provide either --data_path or --dgt_jsonl")

    if not items:
        raise RuntimeError("No training items loaded.")
    
    if args.shuffle:
        random.shuffle(items)

    # Load graph
    gm = GraphMemory.load(args.graph_path)
    nodes = getattr(gm, "nodes", None)
    if not nodes:
        raise RuntimeError(f"Graph has no nodes: {args.graph_path}")

    # Build GT mapping
    id2nodes = build_gt_index(gm)

    # Build retriever
    cfg = RLRetrieverConfig(
        k_demos=int(args.k),
        seed_top_m=int(args.seed_top_m),
        expand_hops=int(args.expand_hops),
        lr=float(args.lr),
        entropy_bonus=float(args.entropy_bonus),
        temperature=float(args.temperature),
        # by stochsstic actions during the RL training the policy can exlore and learn
        greedy_inference=False, 
    )
    retriever = RLRetriever(gm, config=cfg, rng_seed=int(args.seed))

    # Training loop 
    total = len(items)
    start = max(0, int(args.start))
    end = min(total, start + max(0, int(args.num_train)))
    # if start >= end:
    #     raise RuntimeError(f"Nothing to train: start={start}, end={end}, total={total}")

    hits = 0
    # for RL Update with shaped reward
    reward_sum = 0.0
    # missing_gt = 0
    n_used = 0 
    t0 = time.time()

    for step, idx in enumerate(range(start, end), start=1):
        x = items[idx]
        qid = _get_id(x, idx)
        q = _get_question(x)

        if not q:
            # missing_gt += 1
            continue

        gt_nodes = id2nodes.get(str(qid), [])

        if not gt_nodes:
            # missing_gt += 1
            continue

        selected = retriever.select(q, k=int(args.k))

        h = hit_at_k(selected, gt_nodes)
        hits += h
        n_used += 1

        # reward = 1.0 if h == 1 else float(args.reward_neg)
        reward = _compute_shaped_reward(
            gm=gm,
            selected=selected,
            gt_nodes=gt_nodes,
            reward_neg=float(args.reward_neg),
        )
        reward_sum += reward
        retriever.update(reward)

        if step % 50 == 0 and n_used > 0:
            hr = hits / n_used
            avg_reward = reward_sum / n_used
            print(f"[{step}/{end-start}] used={n_used} hit@{args.k}={hr:.3f} avg_reward={avg_reward:.3f}")

        if args.save_policy and (step % max(1, int(args.save_every)) == 0):
            save_policy(retriever, args.save_policy)

    if args.save_policy:
        save_policy(retriever, args.save_policy)

    dt = time.time() - t0
    hr = hits / max(1, n_used)
    avg_reward = reward_sum / max(1, n_used)

    print("\n========== RL TRAIN DONE ==========")
    print(f"graph_path: {args.graph_path}")
    print(f"data_path:  {args.data_path}")
    # print(f"trained_on: {end-start} (skipped_missing_gt={missing_gt})")
    print(f"trained_samples: {n_used}")
    print(f"final hit@{args.k}: {hr:.4f}")
    print(f"avg_reward: {avg_reward:.4f}")
    print(f"time_sec: {dt:.2f}")
    print("========================================\n")


if __name__ == "__main__":
    main()