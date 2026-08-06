from __future__ import annotations
import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple

def _extract_string(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.JoinedStr):
        out: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            else:
                out.append("{expr}")
        return "".join(out)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l = _extract_string(node.left)
        r = _extract_string(node.right)
        if l is not None and r is not None:
            return l + r

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        base = _extract_string(node.func.value)
        if base is not None:
            return base

    return None

def _clean_str(s: str) -> str:
    return s.strip().strip("'").strip('"').strip()

def _strip_optional_table_prefix(s: str) -> str:
    if not s:
        return s
    if "," in s:
        return s.split(",", 1)[1].strip()
    return s.strip()

def _extract_filter_columns_from_string(filter_string: str) -> Set[str]:
    cols: Set[str] = set()
    s = _strip_optional_table_prefix(filter_string)
    parts = s.split("||")

    for p in parts:
        p = p.strip()
        if not p:
            continue

        m = re.search(r"(min|max)\(\s*([A-Za-z0-9_\.]+)\s*\)", p, flags=re.IGNORECASE)
        if m:
            cols.add(m.group(2).split(".")[-1])
            continue

        m = re.search(r"^\s*([A-Za-z0-9_\.]+)\s+is\s+(not\s+)?null\s*$", p, flags=re.IGNORECASE)
        if m:
            cols.add(m.group(1).split(".")[-1])
            continue

        m = re.search(r"^\s*([A-Za-z0-9_\.]+)\s+between\s+", p, flags=re.IGNORECASE)
        if m:
            cols.add(m.group(1).split(".")[-1])
            continue

        m = re.search(r"^\s*([A-Za-z0-9_\.]+)\s+like\s+", p, flags=re.IGNORECASE)
        if m:
            cols.add(m.group(1).split(".")[-1])
            continue

        if re.search(r"\s+in\s+", p, flags=re.IGNORECASE):
            left = re.split(r"\s+in\s+", p, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if left:
                cols.add(left.split(".")[-1])
            continue

        for op in [">=", "<=", "!=", ">", "<", "="]:
            if op in p:
                left = p.split(op, 1)[0].strip()
                if left:
                    cols.add(left.split(".")[-1])
                break

    return cols

def _get_first_colname_from_getvalue_arg(arg: ast.AST) -> Optional[str]:
    s = _extract_string(arg)
    if not s:
        return None
    s = _clean_str(s)
    parts = [x.strip() for x in s.split(",")]
    if not parts:
        return None
    first = parts[0].strip()
    if not first:
        return None
    return first.split(".")[-1].strip() or None

def _get_dict_keys(node: ast.AST) -> List[str]:
    if not isinstance(node, ast.Dict):
        return []
    keys: List[str] = []
    for k in node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.append(k.value)
    return keys

# things i did for the new framework
def _extract_code_from_markdown(code: str) -> str:
    if "```" not in code:
        return code

    blocks = re.findall(r"```(?:python)?\s*(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1]
    return code

DEFAULT_TOOLS = {
    "LoadDB",
    "FilterDB",
    "GetValue",
    "SQLInterpreter",
    "Calendar",
    "Calculate",
    "JoinDB",
    "Count",
    "Sum",
    "Max",
    "Min",
    "Mean",
}

def parse_code_cell(code: str, tool_names: Optional[Set[str]] = None) -> Dict[str, Any]:
    tool_names = tool_names or DEFAULT_TOOLS

    code = _extract_code_from_markdown(code)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "plan_skeleton": "None",
            "schema_footprint": {"tables": [], "columns": []},
        }

    tool_calls: List[Tuple[int, int, str, Dict[str, Any]]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        fn = node.func
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        else:
            continue

        if name not in tool_names:
            continue

        lineno = getattr(node, "lineno", 10**9)
        col = getattr(node, "col_offset", 10**9)
        payload: Dict[str, Any] = {}

        if name == "LoadDB":
            if node.args:
                tbl = _extract_string(node.args[0])
                if tbl:
                    payload["table"] = _clean_str(tbl)

        elif name == "FilterDB":
            if len(node.args) >= 2:
                arg2 = node.args[1]
                if isinstance(arg2, ast.Dict):
                    payload["columns"] = [c.split(".")[-1] for c in _get_dict_keys(arg2)]
                else:
                    s = _extract_string(arg2)
                    if s:
                        payload["columns"] = sorted(_extract_filter_columns_from_string(s))

        elif name == "GetValue":
            if len(node.args) >= 2:
                colname = _get_first_colname_from_getvalue_arg(node.args[1])
                if colname:
                    payload["columns"] = [colname]

        elif name == "JoinDB":
            if len(node.args) >= 3:
                key = _extract_string(node.args[2])
                if key:
                    payload["columns"] = [_clean_str(key).split(".")[-1]]

        tool_calls.append((lineno, col, name, payload))

    tool_calls.sort(key=lambda x: (x[0], x[1]))

    plan_seq = [t[2] for t in tool_calls]
    plan_skeleton = " -> ".join(plan_seq) if plan_seq else "None"

    tables: Set[str] = set()
    columns: Set[str] = set()

    for _, _, tool, payload in tool_calls:
        if tool == "LoadDB" and "table" in payload:
            tables.add(str(payload["table"]))

        cols = payload.get("columns")
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, str) and c.strip():
                    columns.add(c.strip())

    return {
        "plan_skeleton": plan_skeleton,
        "schema_footprint": {
            "tables": sorted(tables),
            "columns": sorted(columns),
        },
    }
