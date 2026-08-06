from __future__ import annotations
import os
import json
import argparse
import re
from typing import Any, Dict, List, Union
from collections import Counter, defaultdict

_TRUE_SET = {"true", "yes", "y", "1"}
_FALSE_SET = {"false", "no", "n", "0", "none", "null", "nan", ""}


def _norm_scalar(x: Any) -> str:
    if x is None:
        return "0"
    s = str(x).strip()

    if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        s = s[1:-1].strip()

    low = s.lower()
    if low in _TRUE_SET:
        return "1"
    if low in _FALSE_SET:
        return "0"

    if re.fullmatch(r"-?\d+\.0", s):
        return s[:-2]

    return s


def _normalize_pred_text(pred: str) -> str:
    if pred is None:
        return ""
    s = str(pred)

    s = re.sub(r"\bTrue\b", "1", s)
    s = re.sub(r"\bFalse\b", "0", s)
    s = re.sub(r"\btrue\b", "1", s)
    s = re.sub(r"\bfalse\b", "0", s)
    s = re.sub(r"\byes\b", "1", s, flags=re.IGNORECASE)
    s = re.sub(r"\bno\b", "0", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnone\b", "0", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnull\b", "0", s, flags=re.IGNORECASE)

    return s


def _normalize_gold(ans: Union[str, List[str], Any]) -> List[str]:
    if isinstance(ans, list):
        return [_norm_scalar(x) for x in ans]
    if isinstance(ans, (int, float)):
        return [_norm_scalar(ans)]
    if isinstance(ans, str):
        g = ans.strip()
        if ", " in g:
            return [_norm_scalar(p) for p in g.split(", ")]
        return [_norm_scalar(g)]
    return [_norm_scalar(ans)]


def rule_judge(pred: str, ans: Union[str, List[str], Any]) -> bool:
    pred_n = _normalize_pred_text(pred)
    gold_list = _normalize_gold(ans)

    for a in gold_list:
        if a not in pred_n:
            return False
    return True


_MARK_TERM = "TERMINATE"
_MARK_EXEC = "[EXECUTION_RESULT]"


def _has_terminate(logs: str) -> bool:
    return _MARK_TERM in logs


def _extract_pred_from_log(logs: str) -> str:
    if _MARK_EXEC in logs:
        return logs.rsplit(_MARK_EXEC, 1)[-1].strip()
    return logs.strip()


def _classify_error(logs: str) -> str:
    if "Traceback (most recent call last)" in logs:
        for t in ["KeyError", "TypeError", "ValueError", "ImportError"]:
            if t in logs:
                return t
        return "Traceback"
    if "rate_limit_exceeded" in logs or "Rate limit" in logs:
        return "RateLimit"
    if "Error:" in logs:
        return "Error"
    return "NoError"


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--id_to_level", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--show_examples", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.logs_path):
        raise RuntimeError(f"--logs_path is not a directory: {args.logs_path}")

    files = sorted([f for f in os.listdir(args.logs_path) if f.endswith(".txt")])

    with open(args.data_path, "r", encoding="utf-8") as f:
        contents = json.load(f)

    # TREQS uses key instead of id
    answers: Dict[str, Any] = {
        str(r["key"]): r["answer"]
        for r in contents
        if "key" in r and "answer" in r
    }

    with open(args.id_to_level, "r", encoding="utf-8") as f:
        id2lvl = {str(k): v for k, v in json.load(f).items()}

    stats = dict(total=0, finished=0, unfinished=0, correct=0, incorrect=0, missing_gold=0)
    error_counter = Counter()
    lvl_stats = defaultdict(lambda: dict(total=0, finished=0, unfinished=0, correct=0, incorrect=0))

    processed = 0
    shown = 0

    for file in files:
        qid = file.split(".")[0]

        if qid not in answers:
            stats["missing_gold"] += 1
            continue

        log_path = os.path.join(args.logs_path, file)
        with open(log_path, "r", encoding="utf-8") as f:
            logs = f.read()

        lvl = id2lvl.get(qid, "UNK")

        stats["total"] += 1
        lvl_stats[lvl]["total"] += 1
        error_counter[_classify_error(logs)] += 1

        if not _has_terminate(logs):
            stats["unfinished"] += 1
            lvl_stats[lvl]["unfinished"] += 1
            processed += 1
            if args.limit != -1 and processed >= args.limit:
                break
            continue

        stats["finished"] += 1
        lvl_stats[lvl]["finished"] += 1

        pred = _extract_pred_from_log(logs)
        ok = rule_judge(pred, answers[qid])

        if ok:
            stats["correct"] += 1
            lvl_stats[lvl]["correct"] += 1
        else:
            stats["incorrect"] += 1
            lvl_stats[lvl]["incorrect"] += 1

        if args.show_examples and shown < args.show_examples:
            print("\n---------------- EXAMPLE ----------------")
            print(f"id: {qid} | level: {lvl} | ok: {ok}")
            print(f"gold: {answers[qid]}")
            print(f"pred: {pred[:800]}{'...' if len(str(pred)) > 800 else ''}")
            print("----------------------------------------")
            shown += 1

        processed += 1
        if args.limit != -1 and processed >= args.limit:
            break

    cr = _safe_div(stats["finished"], stats["total"])
    sr = _safe_div(stats["correct"], stats["total"])

    print("\n==================== OVERALL ====================")
    print(f"total:     {stats['total']}")
    print(f"finished:  {stats['finished']}")
    print(f"unfinished:{stats['unfinished']}")
    print(f"correct:   {stats['correct']}")
    print(f"incorrect: {stats['incorrect']}")
    print(f"CR: {cr:.4f}")
    print(f"SR: {sr:.4f}")
    print("=================================================\n")

    order = ["I", "II", "III", "UNK"]
    print("============== BY COMPLEXITY LEVEL ==============")
    print(f"{'Level':<6} {'Total':>6} {'CR':>8} {'SR':>8} {'Finished':>9} {'Correct':>8}")
    for lvl in order:
        s = lvl_stats.get(lvl)
        if not s or s["total"] == 0:
            continue
        cr_l = _safe_div(s["finished"], s["total"])
        sr_l = _safe_div(s["correct"], s["total"])
        print(f"{lvl:<6} {s['total']:>6} {cr_l:>8.4f} {sr_l:>8.4f} {s['finished']:>9} {s['correct']:>8}")
    print("=================================================\n")

    print("---------- Error-type breakdown (log text) --------")
    for k, v in error_counter.most_common():
        print(f"{k}: {v}")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    main()
