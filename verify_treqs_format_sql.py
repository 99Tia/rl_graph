import json
import re
from collections import Counter

DATA_PATH = "/home/ib5539/code/ehr-rlgraph/data/ehrsql-ehr_rlgraph/treqs/test_preprocessed.jsonl"

TABLE_MAP = {
    0: "DEMOGRAPHIC",
    1: "DIAGNOSES",
    2: "PROCEDURES",
    3: "PRESCRIPTIONS",
    4: "LAB",
}

COLUMN_MAP = {
    "DEMOGRAPHIC": [
        "HADM_ID","SUBJECT_ID","NAME","MARITAL_STATUS","AGE",
        "DOB","GENDER","LANGUAGE","RELIGION","ADMISSION_TYPE",
        "DAYS_STAY","INSURANCE","ETHNICITY","EXPIRE_FLAG",
        "ADMISSION_LOCATION","DISCHARGE_LOCATION","DIAGNOSIS",
        "DOD","DOB_YEAR","DOD_YEAR","ADMITTIME","DISCHTIME","ADMITYEAR"
    ],
    "DIAGNOSES": [
        "SUBJECT_ID","HADM_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"
    ],
    "PROCEDURES": [
        "SUBJECT_ID","HADM_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"
    ],
    "PRESCRIPTIONS": [
        "SUBJECT_ID","HADM_ID","ICUSTAY_ID","DRUG_TYPE","DRUG",
        "FORMULARY_DRUG_CD","ROUTE","DRUG_DOSE"
    ],
    "LAB": [
        "SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","FLAG",
        "VALUE_UNIT","LABEL","FLUID","CATEGORY"
    ],
}

SEL_MAP = {
    0: "",
    1: "COUNT",
    2: "MAX",
    3: "MIN",
    4: "AVG",
}

def normalize_sql(sql: str) -> str:
    sql = (sql or "").upper()
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql

def extract_sql_tables(sql: str):
    sql = normalize_sql(sql)
    found = set()
    for t in TABLE_MAP.values():
        if re.search(rf"\b{re.escape(t)}\b", sql):
            found.add(t)
    return found

def extract_sql_agg(sql: str):
    sql = normalize_sql(sql)
    for agg in ["COUNT", "AVG", "MAX", "MIN"]:
        if re.search(rf"\b{agg}\s*\(", sql):
            return agg
    return ""

def sql_mentions_column(sql: str, table_name: str, col_name: str):
    sql = normalize_sql(sql)
    patterns = [
        rf'{re.escape(table_name)}\."{re.escape(col_name)}"',
        rf'{re.escape(table_name)}\.{re.escape(col_name)}',
        rf'"{re.escape(col_name)}"',
        rf'\b{re.escape(col_name)}\b',
    ]
    return any(re.search(p, sql) for p in patterns)

table_mismatch = []
cond_mismatch = []
agg_mismatch = []

table_ok = 0
cond_ok = 0
agg_ok = 0

table_total = 0
cond_total = 0
agg_total = 0

table_counter = Counter()
cond_counter = Counter()
agg_counter = Counter()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line_id, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)
        qid = row.get("key", row.get("id", f"line_{line_id}"))
        fmt = row.get("format", {}) or {}
        sql = row.get("sql", "")
        sql_tables = extract_sql_tables(sql)
        sql_agg = extract_sql_agg(sql)

        # 1) table check
        fmt_tables = fmt.get("table", []) or []
        fmt_table_names = set(TABLE_MAP[t] for t in fmt_tables if t in TABLE_MAP)
        table_total += 1
        if fmt_table_names == sql_tables:
            table_ok += 1
        else:
            table_mismatch.append({
                "id": qid,
                "question": row.get("question_refine", ""),
                "fmt_tables": sorted(fmt_table_names),
                "sql_tables": sorted(sql_tables),
                "sql": sql,
            })
            table_counter[(tuple(sorted(fmt_table_names)), tuple(sorted(sql_tables)))] += 1

        # 2) condition-column check
        conds = fmt.get("cond", []) or []
        for cond in conds:
            if len(cond) != 4:
                continue

            table_id, col_id, op_id, value = cond
            if table_id not in TABLE_MAP:
                continue

            table_name = TABLE_MAP[table_id]
            if col_id < 0 or col_id >= len(COLUMN_MAP[table_name]):
                continue

            col_name = COLUMN_MAP[table_name][col_id]
            cond_total += 1

            if sql_mentions_column(sql, table_name, col_name):
                cond_ok += 1
            else:
                cond_mismatch.append({
                    "id": qid,
                    "question": row.get("question_refine", ""),
                    "expected_table": table_name,
                    "expected_column": col_name,
                    "value": value,
                    "sql": sql,
                })
                cond_counter[(table_name, col_name)] += 1

        # 3) aggregation check
        sel = fmt.get("sel", 0)
        expected_agg = SEL_MAP.get(sel, "")
        agg_total += 1
        if expected_agg == sql_agg:
            agg_ok += 1
        else:
            agg_mismatch.append({
                "id": qid,
                "question": row.get("question_refine", ""),
                "expected_agg": expected_agg,
                "sql_agg": sql_agg,
                "sql": sql,
            })
            agg_counter[(expected_agg, sql_agg)] += 1

print("\n========== FORMAT ↔ SQL VERIFICATION ==========")
print(f"Table match: {table_ok}/{table_total} = {table_ok / max(1, table_total):.4f}")
print(f"Cond-column match: {cond_ok}/{cond_total} = {cond_ok / max(1, cond_total):.4f}")
print(f"Agg match: {agg_ok}/{agg_total} = {agg_ok / max(1, agg_total):.4f}")

print("\n---- Top table mismatches ----")
for k, v in table_counter.most_common(10):
    print(v, k)

print("\n---- Top cond mismatches ----")
for k, v in cond_counter.most_common(10):
    print(v, k)

print("\n---- Top agg mismatches ----")
for k, v in agg_counter.most_common(10):
    print(v, k)

print("\n---- Sample table mismatches ----")
for x in table_mismatch[:10]:
    print(json.dumps(x, ensure_ascii=False, indent=2))

print("\n---- Sample cond mismatches ----")
for x in cond_mismatch[:10]:
    print(json.dumps(x, ensure_ascii=False, indent=2))

print("\n---- Sample agg mismatches ----")
for x in agg_mismatch[:10]:
    print(json.dumps(x, ensure_ascii=False, indent=2))
