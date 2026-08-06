from __future__ import annotations
import json
import re
import traceback
from typing import Any, Dict, List, Optional
import autogen
from tools import tabtools, calculator


def _normalize_dataset(dataset: Optional[str]) -> str:
    ds = (dataset or "mimic_iii").lower().strip()
    if ds in {"mimic", "mimiciii", "mimic_iii"}:
        return "mimic_iii"
    if ds in {"eicu", "eicu_crd"} or ds.startswith("eicu"):
        return "eicu"
    if ds in {"treqs", "treq"}:
        return "treqs"
    return "mimic_iii"

def _get_code_header(dataset: Optional[str]) -> str:
    ds = _normalize_dataset(dataset)
    if ds == "mimic_iii":
        from ehr_rlgraph.prompts_mimic import CodeHeader
        return CodeHeader
    elif ds == "eicu":
        from ehr_rlgraph.prompts_eicu import CodeHeader
        return CodeHeader
    else:
        from ehr_rlgraph.prompts_treqs import CodeHeader
        return CodeHeader

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()

def _try_parse_json_args(s: str) -> Optional[dict]:
    if not isinstance(s, str):
        return None
    s2 = s.strip()
    if not (s2.startswith("{") and s2.endswith("}")):
        return None
    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

# for the new framework
def format_exec_error(e: BaseException) -> str:
    tb = traceback.format_exc()
    etype = type(e).__name__
    return f"Error: {etype}: {e}\n{tb}"


# need some changes here
def _repair_common_generation_bugs(code: str) -> str:
    # code = code.replace("==", "=")

    if not isinstance(code, str) or not code:
        return code

    def _fix_one_elem_list(m: re.Match) -> str:
        col = m.group(1)
        val = m.group(2)
        return f"{col}={val}"

    code = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*\b)\s*=\s*\[\s*'([^']+)'\s*\]",
        lambda m: _fix_one_elem_list(m),
        code,
    )
    code = re.sub(
        r"(\b[A-Za-z_][A-Za-z0-9_]*\b)\s*=\s*\[\s*\"([^\"]+)\"\s*\]",
        lambda m: _fix_one_elem_list(m),
        code,
    )
    code = re.sub(
        r"(\b[A-Za-z_][A-Za-z0-9_]*\b)\s*=\s*\[\s*([0-9]+)\s*\]",
        lambda m: _fix_one_elem_list(m),
        code,
    )
    code = re.sub(r"OUTTIME\s*=\s*None", "OUTTIME=nan", code, flags=re.IGNORECASE)
    code = re.sub(r"DEATHTIME\s*=\s*None", "DEATHTIME=nan", code, flags=re.IGNORECASE)
    code = re.sub(r"LoadDB\(\s*['\"]inputevents['\"]\s*\)", "LoadDB('inputevents_cv')", code)
    code = re.sub(r"GetValue\(([^,]+),\s*['\"]VOLUME(['\"])(.*?)\2\)", r"GetValue(\1, 'AMOUNT\3')", code)
    code = re.sub(r"VOLUME\s*=", "AMOUNT=", code)
    # code = re.sub(
    #     r"GetValue\(\s*([^,]+)\s*,\s*['\"]VOLUME\s*(,[^'\"]*)?['\"]\s*\)",
    #     lambda m: f"GetValue({m.group(1).strip()}, 'AMOUNT{m.group(2) or ''}')",
    #     code,
    # )
    return code

def run_code(cell: str, dataset: str = "mimic_iii", **_ignored_kwargs) -> str:
    ds = _normalize_dataset(dataset)
    code_header = _get_code_header(ds)

    parsed = _try_parse_json_args(cell) if isinstance(cell, str) else None
    if parsed and ("cell" in parsed or "code" in parsed):
        cell = parsed.get("cell") or parsed.get("code") or ""

    cell = _strip_code_fences(cell)

    if not isinstance(cell, str) or not cell.strip():
        return "Empty code cell. Please provide Python code that assigns final result to variable `answer`."

    cell = _repair_common_generation_bugs(cell)

    env: Dict[str, Any] = {"__builtins__": __builtins__}
    # --- CHANGE: bind dataset-aware tool wrappers ---
    # These wrappers ensure eICU uses eicu.db + eicu csvs, and MIMIC uses mimic_iii.db + mimic csvs.
    # Calculate = calculator.WolframAlphaCalculator
    # LoadDB = lambda name: tabtools.db_loader(name, dataset=ds)
    # FilterDB = tabtools.data_filter
    # GetValue = tabtools.get_value
    # SQLInterpreter = lambda q: tabtools.sql_interpreter(q, dataset=ds)
    # Calendar = lambda dur: tabtools.date_calculator(dur, dataset=ds)

    # env.update({
    #     "Calculate": Calculate,
    #     "LoadDB": LoadDB,
    #     "FilterDB": FilterDB,
    #     "GetValue": GetValue,
    #     "SQLInterpreter": SQLInterpreter,
    #     "Calendar": Calendar,
    # })
    # ----------------------------------
    try:
        # code = code_header.rstrip() + "\n\n" + cell.lstrip()
        # exec(code, env, env)

        exec(code_header, env, env)
        env.update({
            "Calculate": calculator.WolframAlphaCalculator,
            "LoadDB": lambda name: tabtools.db_loader(name, dataset=ds),
            "FilterDB": tabtools.data_filter,
            "GetValue": tabtools.get_value,
            "SQLInterpreter": lambda q: tabtools.sql_interpreter(q, dataset=ds),
            "Calendar": lambda dur: tabtools.date_calculator(dur, dataset=ds),
        })
        exec(cell, env, env)

        if "answer" in env:
            try:
                return str(env["answer"])
            except Exception:
                return repr(env["answer"])

        return "No variable named 'answer' was produced. Please assign the final result to `answer`."

    except Exception as e:
        return format_exec_error(e) + "\nPlease modify the code and make sure the rest works with the modification."

def llm_agent(config_list: List[Dict[str, Any]], system_message: Optional[str] = None):
    if system_message is None:
        system_message = (
            "You are solving EHR QA by writing Python code using ONLY the provided functions:\n"
            "LoadDB, FilterDB, GetValue, SQLInterpreter, Calendar, Calculate.\n"
            "You MUST call the python tool with JSON arguments like {\"cell\": \"...\"}.\n"
            "The code MUST assign the final result to variable `answer`.\n"
            "After the tool returns, reply TERMINATE."
        )

    llm_config = {
        "functions": [
            {
                "name": "python",
                "description": "Execute the provided Python code and return the execution result (answer or error trace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {"type": "string", "description": "Valid Python code to execute."},
                        "code": {"type": "string", "description": "Alias of cell."},
                        "dataset": {"type": "string", "description": "Dataset name: mimic_iii, eicu, or treqs."},
                    },
                    "required": [],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "temperature": 0.0,
    }

    chatbot = autogen.agentchat.AssistantAgent(
        name="chatbot",
        system_message=system_message,
        llm_config=llm_config,
    )
    return chatbot
