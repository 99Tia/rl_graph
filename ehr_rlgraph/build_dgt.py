from __future__ import annotations
import os
import json
import time
import re
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import autogen
from ehr_rlgraph.medagent import MedAgent
from ehr_rlgraph.toolset_high import run_code
from ehr_rlgraph.config import get_backend, llm_config_list
from ehr_rlgraph.memory_graph import GraphMemory

SEP = "\n----------------------------------------------------------\n"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rule_judge(pred: str, ans: Any) -> bool:
    pred = str(pred)
    old_flag = True
    if str(ans) not in pred:
        old_flag = False

    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")

    gold = ans
    if isinstance(gold, str):
        if gold in ["False", "false", "No", "no", "None", "none"]:
            gold = "0"
        if gold in ["True", "true", "Yes", "yes"]:
            gold = "1"
        if ", " in gold:
            gold = gold.split(", ")
        if gold.endswith(".0"):
            gold = gold[:-2]

    if not isinstance(gold, list):
        gold = [gold]

    new_flag = True
    for a in gold:
        if str(a) not in pred:
            new_flag = False
            break

    return (old_flag or new_flag)

# for the TREQS dataset
# def _normalize_scalar_text(x: Any) -> str:
#     s = str(x).strip()

#     if s in ["True", "true", "Yes", "yes"]:
#         s = "1"
#     elif s in ["False", "false", "No", "no", "None", "none"]:
#         s = "0"

#     if s.endswith(".0"):
#         s = s[:-2]

#     return s.strip()


# def _normalize_gold_list(ans: Any) -> List[str]:
#     gold = ans

#     if isinstance(gold, str):
#         gold = _normalize_scalar_text(gold)
#         if ", " in gold:
#             gold = gold.split(", ")
#         else:
#             gold = [gold]
#     elif not isinstance(gold, list):
#         gold = [_normalize_scalar_text(gold)]
#     else:
#         gold = [_normalize_scalar_text(x) for x in gold]

#     out: List[str] = []
#     seen = set()
#     for x in gold:
#         if x not in seen:
#             out.append(x)
#             seen.add(x)
#     return out


# def _extract_pred_items(pred: str) -> List[str]:
#     s = str(pred).strip()
#     tokens = set()

#     for m in re.findall(r"""['"]([^'"]+)['"]""", s):
#         mm = _normalize_scalar_text(m)
#         if mm:
#             tokens.add(mm)

#     for m in re.findall(r"[A-Za-z0-9_./%+-]+", s):
#         mm = _normalize_scalar_text(m)
#         if mm:
#             tokens.add(mm)

#     return list(tokens)


# def rule_judge(pred: str, ans: Any, dataset: str = "mimic_iii") -> bool:
#     if dataset != "treqs":
#         pred = str(pred)
#         old_flag = True
#         if str(ans) not in pred:
#             old_flag = False

#         if "True" in pred:
#             pred = pred.replace("True", "1")
#         else:
#             pred = pred.replace("False", "0")

#         gold = ans
#         if isinstance(gold, str):
#             if gold in ["False", "false", "No", "no", "None", "none"]:
#                 gold = "0"
#             if gold in ["True", "true", "Yes", "yes"]:
#                 gold = "1"
#             if ", " in gold:
#                 gold = gold.split(", ")
#             if gold.endswith(".0"):
#                 gold = gold[:-2]

#         if not isinstance(gold, list):
#             gold = [gold]

#         new_flag = True
#         for a in gold:
#             if str(a) not in pred:
#                 new_flag = False
#                 break

#         return (old_flag or new_flag)

#     pred_str = str(pred)
#     gold_list = _normalize_gold_list(ans)

#     if not gold_list:
#         return False

#     pred_items = set(_extract_pred_items(pred_str))
#     gold_items = set(gold_list)

#     if all(g in pred_str for g in gold_list):
#         return True

#     if gold_items.issubset(pred_items):
#         return True

#     if len(gold_items) == 1:
#         g = next(iter(gold_items))
#         if g in pred_str or g in pred_items:
#             return True

#     return False


def _extract_pred_from_chat(chat_msgs: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    last_cell = ""
    last_fn_out = ""
    last_assistant = ""

    for m in chat_msgs:
        role = m.get("role")

        if role == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                last_assistant = content.strip()

            fc = m.get("function_call")
            if isinstance(fc, dict):
                args = fc.get("arguments")
                if isinstance(args, dict) and isinstance(args.get("cell"), str):
                    last_cell = args["cell"]
                elif isinstance(args, str) and args.strip():
                    last_cell = args

            tcs = m.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                for tc in tcs:
                    fn = (tc.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, dict) and isinstance(args.get("cell"), str):
                        last_cell = args["cell"]
                    elif isinstance(args, str) and args.strip():
                        last_cell = args

        elif role == "function":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                last_fn_out = content.strip()

    pred_text = last_fn_out if last_fn_out else last_assistant
    return pred_text, last_cell, last_fn_out

def _read_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    if raw[0] == "[":
        return json.loads(raw)

    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
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

# for treqs dataset
def _get_id(x: Dict[str, Any], fallback: int) -> str:
    v = x.get("id", None)
    if v is None:
        v = x.get("key", None)
    if v is None:
        v = fallback
    return str(v)


def _get_gold(x: Dict[str, Any]) -> Any:
    return x.get("answer", None)


def _init_seed_memory(dataset: str) -> List[Dict[str, Any]]:
    if dataset == "mimic_iii":
        from ehr_rlgraph.prompts_mimic import EHRAgent_4Shots_Knowledge
    elif dataset == "eicu":
        from ehr_rlgraph.prompts_eicu import EHRAgent_4Shots_Knowledge
    else:
        from ehr_rlgraph.prompts_treqs import EHRAgent_4Shots_Knowledge

    mem: List[Dict[str, Any]] = []
    blocks = EHRAgent_4Shots_Knowledge.split("\n\n")
    for b in blocks:
        b = b.split("Question:")[-1]
        q = b.split("\nKnowledge:\n")[0].strip()
        rest = b.split("\nKnowledge:\n")[-1]
        k = rest.split("\nSolution:")[0].strip()
        c = rest.split("\nSolution:")[-1].strip()
        if q and c:
            mem.append({"question": q, "knowledge": k, "code": c})
    return mem


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_and_save_graph(dgt_jsonl_path: str, graph_out_path: str, k_neighbors: int = 5) -> None:
    demos = _read_jsonl(dgt_jsonl_path)
    if not demos:
        print(f"[WARN] No demos found at {dgt_jsonl_path}. Graph not created.")
        return

    gm = GraphMemory(k_neighbors=k_neighbors)

    supports_rebuild = hasattr(gm, "rebuild")
    add_success_has_rebuild_arg = False
    try:
        import inspect

        sig = inspect.signature(gm.add_success_case)
        add_success_has_rebuild_arg = ("rebuild" in sig.parameters)
    except Exception:
        add_success_has_rebuild_arg = False

    if supports_rebuild and add_success_has_rebuild_arg:
        for d in demos:
            gm.add_success_case(
                d.get("question", ""),
                d.get("code", ""),
                # this is for the train-rl.py patch
                 meta={"id": d.get("id", "")},
                rebuild=False,
            )
        gm.rebuild()
    else:
        mem_list = [{"question": d.get("question", ""), "code": d.get("code", "")} for d in demos]
        if hasattr(gm, "build_from_memory_list"):
            gm.build_from_memory_list(mem_list)
        else:
            for d in demos:
                # gm.add_success_case(d.get("question", ""), d.get("code", ""))
                gm.add_success_case(
                    d.get("question", ""),
                    d.get("code", ""),
                    meta={"id": d.get("id", "")},
                )

    os.makedirs(os.path.dirname(graph_out_path), exist_ok=True)
    gm.save(graph_out_path)
    print(f"[INFO] Saved DGT graph -> {graph_out_path} (nodes={len(getattr(gm, 'nodes', []))})")


@dataclass
class BuildDGTConfig:
    dataset: str
    llm: str
    data_path: str
    out_path: str
    graph_out: str
    logs_dir: str
    num_shots: int = 4
    num_examples: int = -1
    start: int = 0
    seed: int = 0
    max_turns: int = 10
    shuffle: bool = True
    only_correct: bool = True
    save_all_attempts: bool = False


def _force_stage1_levenshtein(user_proxy: MedAgent) -> None:
    if hasattr(user_proxy, "set_retrieval_mode") and callable(getattr(user_proxy, "set_retrieval_mode")):
        try:
            user_proxy.set_retrieval_mode("levenshtein")
            return
        except Exception:
            pass

    try:
        setattr(user_proxy, "retrieval_mode", "levenshtein")
    except Exception:
        pass

    for attr in ("graph", "retriever"):
        if hasattr(user_proxy, attr):
            try:
                setattr(user_proxy, attr, None)
            except Exception:
                pass

# def _adapt_treqs_question(q: str) -> str:
#         q0 = (q or "").strip()
#         low = q0.lower()
#         hints = []

#         if any(x in low for x in [
#             "how many", "what is the number", "give the number",
#             "count the number", "count "
#             ]):
#             hints.append("Return one scalar count only, not a list.")

#         if "primary disease" in low:
#             hints.append("For TREQS, primary disease often matches exact DEMOGRAPHIC.DIAGNOSIS.")

#         if "drug code" in low:
#             hints.append("Use exact PRESCRIPTIONS.FORMULARY_DRUG_CD equality.")

#         if "icd9 code" in low:
#             hints.append("Use exact ICD9_CODE equality.")

#         if "lab item id" in low:
#             hints.append("Use exact LAB.ITEMID equality.")

#         if "diagnosed with" in low and "primary disease" not in low:
#             hints.append("Prefer exact DIAGNOSES.SHORT_TITLE or LONG_TITLE equality before fuzzy matching.")

#         if "language" in low or "speak" in low:
#             hints.append("Prefer exact LANGUAGE value from the dataset before fuzzy matching.")

#         if "before the year" in low or "admitted before" in low or "born before" in low or "died in or before" in low:
#             hints.append("Prefer year columns such as ADMITYEAR, DOB_YEAR, or DOD_YEAR.")

#         if "for how long" in low or "duration" in low:
#             hints.append("If the benchmark expects a scalar number, do not return a raw list.")

#         if any(x in low for x in ["diagnosis", "procedure", "lab", "drug", "insurance", "married", "female", "male"]):
#             hints.append("For multi-table count questions, benchmark often uses DEMOGRAPHIC.HADM_ID = OTHER_TABLE.HADM_ID.")

#         if not hints:
#             return q0

#         return q0 + "\n\nTREQS hints:\n- " + "\n- ".join(dict.fromkeys(hints))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", type=str, required=True, help="OpenAI model name, e.g. gpt-4.1")
    p.add_argument("--dataset", type=str, required=True, choices=["mimic_iii", "eicu", "treqs"])
    p.add_argument("--data_path", type=str, required=True, help="Input dataset (json array or jsonl). Must contain answers.")
    p.add_argument("--out_path", type=str, required=True, help="Output JSONL file for D_GT (correct demos).")
    p.add_argument("--graph_out", type=str, required=True, help="Output GraphMemory JSON (built from D_GT).")
    p.add_argument("--logs_dir", type=str, required=True, help="Directory to save per-id logs.")
    p.add_argument("--num_shots", type=int, default=4)
    p.add_argument("--num_examples", type=int, default=-1, help="-1 = all.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_turns", type=int, default=10)
    p.add_argument("--no_shuffle", action="store_true")
    p.add_argument("--save_all_attempts", action="store_true", help="Also save incorrect attempts into <out_path>.bad.jsonl")
    p.add_argument("--keep_incorrect", action="store_true", help="Keep incorrect demos in D_GT (NOT recommended).")
    args = p.parse_args()

    cfg = BuildDGTConfig(
        dataset=args.dataset,
        llm=args.llm,
        data_path=args.data_path,
        out_path=args.out_path,
        graph_out=args.graph_out,
        logs_dir=args.logs_dir,
        num_shots=args.num_shots,
        num_examples=args.num_examples,
        start=args.start,
        seed=args.seed,
        max_turns=args.max_turns,
        shuffle=(not args.no_shuffle),
        only_correct=(not args.keep_incorrect),
        save_all_attempts=args.save_all_attempts,
    )

    set_seed(cfg.seed)

    backend_cfg = get_backend(cfg.llm)
    config_list = [backend_cfg]
    llm_cfg = llm_config_list(cfg.seed, cfg.llm)

    chatbot = autogen.agentchat.AssistantAgent(
        name="chatbot",
        system_message=(
            "You are solving EHR QA by writing python code using the provided helper functions.\n"
            "Write code that assigns the final result to variable `answer`.\n"
            "Call the python tool so the code executes.\n"
            "Reply TERMINATE when finished."
        ),
        llm_config=llm_cfg,
    )

    user_proxy = MedAgent(
        name="user_proxy",
        is_termination_msg=lambda x: isinstance(x.get("content", ""), str)
        and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=cfg.max_turns,
        code_execution_config=False,
        config_list=config_list,
        debug=False,
    )

    _force_stage1_levenshtein(user_proxy)

    def _python_tool(cell: str = "", code: str = "", dataset: str = cfg.dataset, **kwargs):
        src = cell or code or ""
        return run_code(src, dataset=dataset)

    user_proxy.register_function(function_map={"python": _python_tool})
    user_proxy.register_dataset(cfg.dataset)

    items = _read_dataset(cfg.data_path)
    if cfg.shuffle:
        random.shuffle(items)

    long_term_memory: List[Dict[str, Any]] = _init_seed_memory(cfg.dataset)

    run_logs_dir = os.path.join(cfg.logs_dir, f"dgt_{cfg.dataset}_{cfg.llm}_k{cfg.num_shots}")
    os.makedirs(run_logs_dir, exist_ok=True)

    bad_out_path = cfg.out_path + ".bad.jsonl"

    total = len(items)
    start_idx = max(0, cfg.start)
    end_idx = total if cfg.num_examples == -1 else min(total, start_idx + max(0, cfg.num_examples))

    kept = 0
    processed = 0
    t0 = time.time()

    for idx in range(start_idx, end_idx):
        x = items[idx]
        qid = _get_id(x, idx)
        question = _get_question(x)
        # if cfg.dataset == "treqs":
        #     question = _adapt_treqs_question(question)
        gold = _get_gold(x)

        processed += 1

        if gold is None:
            log_text = (
                f"[SKIP_NO_GOLD]\n"
                f"id={qid}\n"
                f"question={question}\n"
                f"reason=missing 'answer' in dataset row\n"
            )
            _write_text(os.path.join(run_logs_dir, f"{qid}.txt"), log_text)
            print(f"[{processed}/{end_idx-start_idx}] id={qid} SKIP (no gold answer)")
            continue

        user_proxy.question = ""
        user_proxy.code = ""
        user_proxy.knowledge = ""

        user_proxy.update_memory(cfg.num_shots, long_term_memory)

        pred_text = ""
        last_cell = ""
        last_fn_out = ""
        chat_dump: List[Dict[str, Any]] = []

        try:
            user_proxy.initiate_chat(chatbot, message=question)

            if hasattr(user_proxy, "chat_messages") and chatbot in user_proxy.chat_messages:
                chat_dump = user_proxy.chat_messages[chatbot]
            else:
                chat_dump = []

            pred_text, last_cell, last_fn_out = _extract_pred_from_chat(chat_dump)

        except Exception as e:
            log_text = (
                f"[EXCEPTION]\n"
                f"id={qid}\n"
                f"question={question}\n"
                f"error={repr(e)}\n"
            )
            _write_text(os.path.join(run_logs_dir, f"{qid}.txt"), log_text)
            print(f"[{processed}/{end_idx-start_idx}] id={qid} EXCEPTION")
            continue

        ok = rule_judge(pred_text, gold)
        # for TREQS dataset
        # ok = rule_judge(pred_text, gold, dataset=cfg.dataset)

        lines: List[str] = []
        lines.append(f"[QUESTION]\n{question}")
        lines.append(f"[GOLD]\n{gold}")
        lines.append(f"[PRED_TEXT]\n{pred_text}")
        if isinstance(last_cell, str) and last_cell.strip():
            lines.append(f"[PYTHON_CELL_OR_ARGS]\n{last_cell}")
        if isinstance(last_fn_out, str) and last_fn_out.strip():
            lines.append(f"[EXECUTION_RESULT]\n{last_fn_out}")
        lines.append(f"[JUDGE]\nrule_ok={ok}")
        lines.append("[FULL_CHAT_DUMP]")
        for m in chat_dump:
            lines.append(json.dumps(m, ensure_ascii=False))
        _write_text(os.path.join(run_logs_dir, f"{qid}.txt"), (SEP.join(lines) + "\n"))

        demo = {
            "id": qid,
            "dataset": cfg.dataset,
            "question": question,
            "knowledge": getattr(user_proxy, "knowledge", ""),
            "code": getattr(user_proxy, "code", ""),
            "gold_answer": gold,
            "pred_text": pred_text,
            "rule_ok": bool(ok),
            "num_shots": cfg.num_shots,
            "llm": cfg.llm,
            "timestamp": time.time(),
        }

        if ok or (not cfg.only_correct):
            _append_jsonl(cfg.out_path, demo)
            kept += 1

            if ok:
                long_term_memory.append(
                    {
                        "question": question,
                        "knowledge": demo["knowledge"],
                        "code": demo["code"],
                    }
                )
        else:
            if cfg.save_all_attempts:
                _append_jsonl(bad_out_path, demo)

        print(f"[{processed}/{end_idx-start_idx}] id={qid} ok={ok} kept={kept} memory={len(long_term_memory)}")

    build_and_save_graph(cfg.out_path, cfg.graph_out, k_neighbors=5)

    print("\n========== BUILD_DGT DONE ==========")
    print(f"dataset: {cfg.dataset}")
    print(f"llm: {cfg.llm}")
    print(f"input: {cfg.data_path}")
    print(f"out_path (D_GT): {cfg.out_path}")
    print(f"graph_out: {cfg.graph_out}")
    if cfg.save_all_attempts:
        print(f"bad_out_path: {bad_out_path}")
    print(f"logs_dir: {run_logs_dir}")
    print(f"processed: {processed}")
    print(f"kept: {kept}")
    print(f"time_sec: {time.time()-t0:.2f}")
    print("===================================\n")


if __name__ == "__main__":
    main()