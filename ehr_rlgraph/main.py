from __future__ import annotations
import os
import json
import random
import numpy as np
import argparse
import time
import re
from typing import Any, Dict, List, Optional
import autogen
from ehr_rlgraph.toolset_high import run_code
from ehr_rlgraph.medagent import MedAgent
from ehr_rlgraph.config import get_backend, llm_config_list
# try:
#     from ehr_rlgraph.memory_graph import GraphMemory
# except Exception:
#     GraphMemory = None
import traceback

def judge(pred: str, ans: Any) -> bool:
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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

def _extract_cell_from_any_msg(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    tcs = msg.get("tool_calls", None)
    if isinstance(tcs, list) and tcs:
        for tc in tcs:
            fn = (tc.get("function") or {})
            raw_args = fn.get("arguments", "")
            if isinstance(raw_args, dict):
                cell = raw_args.get("cell") or raw_args.get("code")
                if isinstance(cell, str) and cell.strip():
                    return cell.strip()
            if isinstance(raw_args, str) and raw_args.strip():
                s = raw_args.strip()
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        cell = parsed.get("cell") or parsed.get("code")
                        if isinstance(cell, str) and cell.strip():
                            return cell.strip()
                except Exception:
                    if "LoadDB" in s or "SQLInterpreter" in s or "answer" in s:
                        return _strip_code_fences(s)
    fc = msg.get("function_call", None)
    if isinstance(fc, dict):
        raw_args = fc.get("arguments", "")
        if isinstance(raw_args, dict):
            cell = raw_args.get("cell") or raw_args.get("code")
            if isinstance(cell, str) and cell.strip():
                return cell.strip()
        if isinstance(raw_args, str) and raw_args.strip():
            s = raw_args.strip()
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    cell = parsed.get("cell") or parsed.get("code")
                    if isinstance(cell, str) and cell.strip():
                        return cell.strip()
            except Exception:
                if "LoadDB" in s or "SQLInterpreter" in s or "answer" in s:
                    return _strip_code_fences(s)
    content = msg.get("content", None)
    if isinstance(content, str) and content.strip():
        m = re.search(r'"cell"\s*:\s*"(.+?)"', content, flags=re.DOTALL)
        if m:
            cell = m.group(1)
            cell = cell.replace(r"\n", "\n").replace(r"\"", '"')
            return cell.strip()
        fenced = _strip_code_fences(content)
        if "LoadDB" in fenced or "SQLInterpreter" in fenced or "answer" in fenced:
            return fenced
    return ""

def _extract_last_execution_result(logs_lines: List[str]) -> str:
    for i in range(len(logs_lines) - 1, -1, -1):
        if logs_lines[i].startswith("[EXECUTION_RESULT]"):
            parts = logs_lines[i].split("\n", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return ""

def _read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# for the new framework
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def _load_dgt_memory(dgt_jsonl: str) -> List[Dict[str, Any]]:
    rows = _read_jsonl(dgt_jsonl)
    mem: List[Dict[str, Any]] = []
    for r in rows:
        q = (r.get("question") or "").strip()
        k = (r.get("knowledge") or "").strip()
        c = (r.get("code") or "").strip()
        if q and c:
            mem.append({"question": q, "knowledge": k, "code": c, "id": r.get("id", None)})
    return mem

def _load_policy_state(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        return obj if isinstance(obj, dict) else None

# def _load_rl_state(path: str) -> Optional[Dict[str, Any]]:
#     if not path:
#         return None
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


def _ensure_retriever_and_load_policy(user_proxy: MedAgent, question: str, policy_state: Optional[Dict[str, Any]]) -> None:
    if policy_state is None:
        return
    if hasattr(user_proxy, "_ensure_graph_and_retriever") and callable(getattr(user_proxy, "_ensure_graph_and_retriever")):
        user_proxy._ensure_graph_and_retriever()
    else:
        try:
            _ = user_proxy.retrieve_examples(question)
        except Exception:
            pass
    retr = getattr(user_proxy, "retriever", None)
    if retr is None:
        print("[WARN] Policy provided but retriever is still None. Check MedAgent graph_path/retrieval_mode.")
        return
    
    try:
        retr.load_state(policy_state)
    except Exception as e:
        print(f"[WARN] Failed to load policy state into retriever: {e}")
        return
    
    try:
        if hasattr(retr, "cfg") and hasattr(retr.cfg, "greedy_inference"):
            retr.cfg.greedy_inference = True
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mimic_iii", choices=["mimic_iii", "eicu", "treqs"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--logs_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--start_id", type=int, default=0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_id", type=str, default="")

    # D_GT (Stage 2 output)
    parser.add_argument("--dgt_jsonl", type=str, default="", help="D_GT jsonl produced by build_dgt.py (correct demos only).")

    # Graph + policy (Stage 5 output)
    parser.add_argument("--graph_path", type=str, default="", help="GraphMemory.save output json (built from D_GT).")
    parser.add_argument("--policy_path", type=str, default="", help="Policy json saved by train_rl.py (RLRetriever.get_state()).")

    parser.add_argument("--shuffle", action="store_true", help="Shuffle evaluation questions (optional).")

    # For ablation study to disable error debugger
    # parser.add_argument("--disable_error_debugger", action="store_true")

    # For ablation study of without knowledge retrieval
    # parser.add_argument("--disable_knowledge", action="store_true")

    args = parser.parse_args()

    # Temporary cap to reduce prompt size / TPM usage
    # if args.num_shots > 2:
    #     args.num_shots = 2
    set_seed(args.seed)

    backend_cfg = get_backend(args.llm)
    config_list = [backend_cfg]
    llm_cfg = llm_config_list(args.seed, args.llm)

    system_message = (
        "For coding tasks, only use the functions you have been provided with. "
        "Reply TERMINATE when the task is done. "
        "Write python code that assigns the final result to variable `answer`."
    )
    chatbot = autogen.agentchat.AssistantAgent(
        name="chatbot",
        system_message=system_message,
        llm_config=llm_cfg,
    )

    user_proxy = MedAgent(
        name="user_proxy",
        is_termination_msg=lambda x: isinstance(x.get("content", ""), str)
        and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=8,
        code_execution_config=False,
        config_list=config_list,
        # For ablation study of without knowledge retrieval
        # use_llm_knowledge=False,
        debug=bool(args.debug),
        graph_path=args.graph_path,
        retrieval_mode=("graph_rl" if args.graph_path else "levenshtein"),
        k_neighbors=5,
        # For ablation study
        # disable_error_debugger=bool(args.disable_error_debugger),
    )
    # user_proxy.register_function(
    #     function_map={"python": lambda cell, dataset=args.dataset: run_code(cell, dataset=dataset)}
    # )
    def _python_tool(cell: str = "", code: str = "", dataset: str = args.dataset, **kwargs):
        src = cell or code or ""
        return run_code(src, dataset=dataset)

    user_proxy.register_function(function_map={"python": _python_tool})

    user_proxy.register_dataset(args.dataset)
    long_term_memory: List[Dict[str, Any]] = []

    if args.dgt_jsonl:
        long_term_memory = _load_dgt_memory(args.dgt_jsonl)
        print(f"[INFO] Loaded D_GT demos for inference: {len(long_term_memory)} from {args.dgt_jsonl}")
    else:
        if args.dataset == "mimic_iii":
            from ehr_rlgraph.prompts_mimic import EHRAgent_4Shots_Knowledge
        elif args.dataset == "eicu":
            from ehr_rlgraph.prompts_eicu import EHRAgent_4Shots_Knowledge
        else:
            from ehr_rlgraph.prompts_treqs import EHRAgent_4Shots_Knowledge
            
        init_memory = EHRAgent_4Shots_Knowledge.split("\n\n")
        for item in init_memory:
            item = item.split("Question:")[-1]
            q = item.split("\nKnowledge:\n")[0].strip()
            rest = item.split("\nKnowledge:\n")[-1]
            knowledge = rest.split("\nSolution:")[0].strip()
            code = rest.split("\nSolution:")[-1].strip()
            if q and code:
                long_term_memory.append({"question": q, "knowledge": knowledge, "code": code})

        print(f"[INFO] Using only prompt bootstrapping memory: {len(long_term_memory)} demos")

    # if args.graph_json:
    #     if GraphMemory is None:
    #         print("[WARN] GraphMemory not importable, ignoring --graph_json.")
    #     else:
    #         try:
    #             gm = GraphMemory.load(args.graph_json)
    #             print(f"[INFO] Loaded GraphMemory from {args.graph_json} with nodes={len(getattr(gm, 'nodes', []))}")
    #         except Exception as e:
    #             print(f"[WARN] Failed to load graph_json: {e}")

    # if args.rl_state:
    #     state = _load_rl_state(args.rl_state)
    #     if state is not None and hasattr(user_proxy, "retriever") and user_proxy.retriever is not None:
    #         user_proxy._pending_rl_state = state  
    #         print(f"[INFO] Will load RL state after retriever init: {args.rl_state}")
    #     else:
    #         user_proxy._pending_rl_state = state  
    #         print(f"[INFO] Will load RL state after retriever init: {args.rl_state}")
            
    # with open(args.data_path, "r", encoding="utf-8") as f:
    #     contents = json.load(f)
    # random.shuffle(contents)
    
    policy_state = _load_policy_state(args.policy_path) if args.policy_path else None
    if args.policy_path and policy_state is None:
        print(f"[WARN] policy_path provided but could not load JSON: {args.policy_path}")


    contents = _read_json(args.data_path)
    if args.shuffle:
        random.shuffle(contents)

    os.makedirs(os.path.join(args.logs_path, str(args.num_shots)), exist_ok=True)
    file_path_tmpl = os.path.join(args.logs_path, str(args.num_shots), "{id}.txt")

    if args.num_questions == -1:
        args.num_questions = len(contents)

    end_idx = min(len(contents), args.start_id + args.num_questions)
    start_time = time.time()

    for i in range(args.start_id, end_idx):
        if args.debug and args.debug_id and str(contents[i].get("id", "")) != str(args.debug_id):
            continue

        # qid = contents[i].get("id", i)
        # for treqs dataset
        qid = contents[i].get("id", contents[i].get("key", i))
        # question = contents[i].get("template") or contents[i].get("question") or ""
        question = (
            contents[i].get("template")
            or contents[i].get("question")
            or contents[i].get("question_refine")
            or contents[i].get("sql")
            or ""
            )
        gold = contents[i].get("answer", None)

        logs_lines: List[str] = []
        pred_text = ""

        try:
            user_proxy.update_memory(args.num_shots, long_term_memory)

            _ensure_retriever_and_load_policy(user_proxy, question, policy_state)

            # pending = getattr(user_proxy, "_pending_rl_state", None)
            # if pending is not None and getattr(user_proxy, "retriever", None) is not None:
            #     try:
            #         user_proxy.retriever.load_state(pending) 
            #         user_proxy._pending_rl_state = None  
            #         print("[INFO] Loaded RL weights into retriever.")
            #     except Exception as e:
            #         print(f"[WARN] Failed to load RL state: {e}")

            user_proxy.initiate_chat(chatbot, message=question)

            logs = user_proxy._oai_messages
            logs_lines.append(f"[QUESTION]\n{question}")
            logs_lines.append(f"[GOLD]\n{gold}")

            last_cell = ""
            last_exec = ""

            for agent in list(logs.keys()):
                for msg in logs[agent]:
                    if isinstance(msg, dict) and msg.get("content") is not None:
                        logs_lines.append(f"[MSG content]\n{str(msg.get('content'))}")

                    cell = _extract_cell_from_any_msg(msg)
                    if cell:
                        last_cell = cell
                        logs_lines.append(f"[PYTHON_CELL]\n{cell}")

                    if isinstance(msg, dict) and msg.get("role") == "function":
                        c = msg.get("content", "")
                        if isinstance(c, str) and c.strip():
                            last_exec = c.strip()
                            logs_lines.append(f"[EXECUTION_RESULT]\n{last_exec}")

            pred_text = last_exec.strip() if last_exec.strip() else last_cell.strip()

        except Exception as e:
            traceback.print_exc()
            logs_lines.append(f"[EXCEPTION]\n{repr(e)}")

        file_path = file_path_tmpl.format(id=qid)
        joined = "\n" + ("\n" + "-" * 70 + "\n").join(logs_lines) + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(joined)

        if not pred_text:
            pred_text = _extract_last_execution_result(logs_lines)

        ok = judge(pred_text, gold)

        if args.debug:
            print(f"[DEBUG] id={qid} ok={ok} pred={pred_text}")

    print("Time elapsed:", time.time() - start_time)


if __name__ == "__main__":
    main()
