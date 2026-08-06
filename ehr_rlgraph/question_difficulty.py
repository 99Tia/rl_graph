from __future__ import annotations
import json
import re
import os
import argparse
import collections
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sns.set_theme(
    style="ticks",
    font="Times New Roman",
    font_scale=2.1,
    rc={"grid.linestyle": ":", "axes.grid": True},
)

# ----------------------------
# Helpers
# ----------------------------
def read_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(x) for x in raw.splitlines() if x.strip()]

def num_q_tag_var(q_tag: str) -> int:
    s = q_tag or ""
    return s.count("{") + s.count("[")

def extract_tables(sql: str):
    sql = sql or ""
    tables = re.findall(r"\bfrom\s+(\w+)\b", sql, flags=re.IGNORECASE)
    joins = re.findall(r"\bjoin\s+(\w+)\b", sql, flags=re.IGNORECASE)
    return list(set(tables + joins))

def extract_columns(sql: str):
    sql = sql or ""
    cols = re.findall(r"\b\w+\.\w+\b", sql)
    cols = [c for c in cols if not re.match(r"^t\d+\.", c)]
    return list(set(cols))

def level_from_qtag_vars(n: int) -> str:
    # Simple rule (your choice):
    # 0 -> I, 1 -> II, 2 -> III, >=3 -> IV
    if n <= 0:
        return "I"
    if n == 1:
        return "II"
    if n == 2:
        return "III"
    return "IV"

def plot_dist(values, xlabel, outpath):
    counter = collections.Counter(values)
    if not counter:
        return
    xs, ys = zip(*sorted(counter.items()))
    plt.figure(figsize=(6, 5.5), dpi=120)
    sns.barplot(x=list(xs), y=list(ys), color=sns.color_palette()[0])
    plt.ylabel("Frequency", size=30)
    plt.xlabel(xlabel, size=30)
    plt.tight_layout(rect=[-0.05, -0.05, 1.05, 1.05])
    plt.savefig(outpath)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mimic_iii", choices=["mimic_iii", "eicu"])
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to valid_preprocessed.json (must include id, q_tag, query).")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory (will be created if missing).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    annotated_out = os.path.join(args.out_dir, f"{args.dataset}_annotated.jsonl")
    id_to_level_path = os.path.join(args.out_dir, "id_to_level.json")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    data = read_json_any(args.data_path)

    list_num_q_tag_var = []
    list_num_tables = []
    list_num_columns = []

    bucket_num_q_tag_var = defaultdict(list)
    bucket_num_tables = defaultdict(list)
    bucket_num_columns = defaultdict(list)
    bucket_level = defaultdict(list)

    id_to_level = {}

    with open(annotated_out, "w", encoding="utf-8") as f_out:
        for x in data:
            qid = str(x.get("id", ""))
            if not qid:
                continue

            qtag = x.get("q_tag", "")
            sql = x.get("query", "")

            n_var = num_q_tag_var(qtag)
            tables = extract_tables(sql)
            cols = extract_columns(sql)

            n_tables = len(tables)
            n_cols = len(cols)

            lvl = level_from_qtag_vars(n_var)

            x["num_q_tag_var"] = n_var
            x["num_tables"] = n_tables
            x["num_columns"] = n_cols
            x["level"] = lvl

            f_out.write(json.dumps(x) + "\n")

            list_num_q_tag_var.append(n_var)
            list_num_tables.append(n_tables)
            list_num_columns.append(n_cols)

            bucket_num_q_tag_var[n_var].append(qid)
            bucket_num_tables[n_tables].append(qid)
            bucket_num_columns[n_cols].append(qid)
            bucket_level[lvl].append(qid)

            id_to_level[qid] = lvl

    # Save id->level mapping (MOST IMPORTANT for per-level evaluation)
    with open(id_to_level_path, "w", encoding="utf-8") as f:
        json.dump(id_to_level, f, indent=2)

    # save buckets
    for subname, bucket in [
        ("num_q_tag_var", bucket_num_q_tag_var),
        ("num_tables", bucket_num_tables),
        ("num_columns", bucket_num_columns),
        ("levels", bucket_level),
    ]:
        subdir = os.path.join(args.out_dir, subname)
        os.makedirs(subdir, exist_ok=True)
        for k, ids in bucket.items():
            with open(os.path.join(subdir, f"{k}.json"), "w") as f:
                json.dump(ids, f)

    # plots
    plot_dist(list_num_q_tag_var, "# q_tag variables", os.path.join(fig_dir, "num_q_tag_var_distri.pdf"))
    plot_dist(list_num_tables, "# tables", os.path.join(fig_dir, "num_tables_distri.pdf"))
    plot_dist(list_num_columns, "# columns", os.path.join(fig_dir, "num_columns_distri.pdf"))

    print("[INFO] Saved out_dir:", args.out_dir)
    print("[INFO] Annotated jsonl:", annotated_out)
    print("[INFO] id_to_level:", id_to_level_path)
    print("[INFO] Level counts:", {k: len(v) for k, v in bucket_level.items()})

if __name__ == "__main__":
    main()
