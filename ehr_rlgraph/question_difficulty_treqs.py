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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sns.set_theme(
    style="ticks",
    font_scale=2.1,
    rc={"grid.linestyle": ":", "axes.grid": True},
)


def read_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(x) for x in raw.splitlines() if x.strip()]


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


def extract_conditions(sql: str):
    sql = sql or ""
    conds = re.findall(r"\bwhere\b(.+)", sql, flags=re.IGNORECASE)
    if not conds:
        return 0
    cond = conds[0]
    return cond.count("=") + cond.lower().count(" like ") + cond.lower().count(" in ")


def level_from_sql_complexity(n_tables, n_cols, n_conds):

    score = n_tables + n_cols + n_conds

    # TREQS has 3 levels
    if score <= 3:
        return "I"
    elif score <= 6:
        return "II"
    else:
        return "III"


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to test_preprocessed.json")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    annotated_out = os.path.join(args.out_dir, "treqs_annotated.jsonl")
    id_to_level_path = os.path.join(args.out_dir, "id_to_level.json")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    data = read_json_any(args.data_path)

    list_num_tables = []
    list_num_columns = []
    list_num_conditions = []

    bucket_tables = defaultdict(list)
    bucket_columns = defaultdict(list)
    bucket_conditions = defaultdict(list)
    bucket_level = defaultdict(list)

    id_to_level = {}

    with open(annotated_out, "w", encoding="utf-8") as f_out:

        for x in data:

            qid = str(x.get("key", ""))  # TREQS uses "key"
            sql = x.get("sql", "")

            tables = extract_tables(sql)
            cols = extract_columns(sql)
            conds = extract_conditions(sql)

            n_tables = len(tables)
            n_cols = len(cols)

            lvl = level_from_sql_complexity(n_tables, n_cols, conds)

            x["num_tables"] = n_tables
            x["num_columns"] = n_cols
            x["num_conditions"] = conds
            x["level"] = lvl

            f_out.write(json.dumps(x) + "\n")

            list_num_tables.append(n_tables)
            list_num_columns.append(n_cols)
            list_num_conditions.append(conds)

            bucket_tables[n_tables].append(qid)
            bucket_columns[n_cols].append(qid)
            bucket_conditions[conds].append(qid)
            bucket_level[lvl].append(qid)

            id_to_level[qid] = lvl

    with open(id_to_level_path, "w") as f:
        json.dump(id_to_level, f, indent=2)

    plot_dist(list_num_tables, "# tables", os.path.join(fig_dir, "num_tables_distri.pdf"))
    plot_dist(list_num_columns, "# columns", os.path.join(fig_dir, "num_columns_distri.pdf"))
    plot_dist(list_num_conditions, "# conditions", os.path.join(fig_dir, "num_conditions_distri.pdf"))

    print("[INFO] Saved:", args.out_dir)
    print("[INFO] Level counts:", {k: len(v) for k, v in bucket_level.items()})


if __name__ == "__main__":
    main()
