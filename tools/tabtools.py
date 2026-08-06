import pandas as pd
import jsonlines
import json
import re
import sqlite3
import sys
import Levenshtein

BASE = "/home/ib5539/code/ehr-rlgraph/data/ehrsql-ehr_rlgraph"


def _normalize_dataset(ds: str) -> str:
    ds = (ds or "mimic_iii").lower().strip()
    if ds in {"mimic", "mimiciii", "mimic_iii"}:
        return "mimic_iii"
    if ds in {"eicu", "eicu_crd"} or ds.startswith("eicu"):
        return "eicu"
    if ds in {"treqs", "treq"}:
        return "treqs"
    return ds


def _normalize_table_name(name: str, dataset: str = "mimic_iii") -> str:
    if not isinstance(name, str):
        return name
    n = name.strip().lower()
    ds = _normalize_dataset(dataset)

    aliases = {
        "admission": "admissions",
        "diagnosis_icd": "diagnoses_icd",
        "procedure_icd": "procedures_icd",
    }

  
    if ds == "mimic_iii":
        aliases.update({
            "inputevents": "inputevents_cv",
            "inputevent": "inputevents_cv",
            "outputevent": "outputevents",
            "outputevents_cv": "outputevents",
            "patient": "patients",  
        })
    elif ds == "eicu":
        aliases.update({
            "patients": "patient",
        })
    elif ds =="treqs":
        aliases.update({
            "demographic": "DEMOGRAPHIC",
            "diagnoses": "DIAGNOSES",
            "lab": "LAB",
            "prescriptions": "PRESCRIPTIONS",
            "procedures": "PROCEDURES",
        })

    return aliases.get(n, n)


def db_loader(target_ehr, dataset: str = "mimic_iii"):
    ds = _normalize_dataset(dataset)

    if ds == "mimic_iii":
        ehr_dict = {
            "admissions": f"{BASE}/mimic_iii/ADMISSIONS.csv",
            "chartevents": f"{BASE}/mimic_iii/CHARTEVENTS.csv",
            "cost": f"{BASE}/mimic_iii/COST.csv",
            "d_icd_diagnoses": f"{BASE}/mimic_iii/D_ICD_DIAGNOSES.csv",
            "d_icd_procedures": f"{BASE}/mimic_iii/D_ICD_PROCEDURES.csv",
            "d_items": f"{BASE}/mimic_iii/D_ITEMS.csv",
            "d_labitems": f"{BASE}/mimic_iii/D_LABITEMS.csv",
            "diagnoses_icd": f"{BASE}/mimic_iii/DIAGNOSES_ICD.csv",
            "icustays": f"{BASE}/mimic_iii/ICUSTAYS.csv",
            "inputevents_cv": f"{BASE}/mimic_iii/INPUTEVENTS_CV.csv",
            "labevents": f"{BASE}/mimic_iii/LABEVENTS.csv",
            "microbiologyevents": f"{BASE}/mimic_iii/MICROBIOLOGYEVENTS.csv",
            "outputevents": f"{BASE}/mimic_iii/OUTPUTEVENTS.csv",
            "patients": f"{BASE}/mimic_iii/PATIENTS.csv",
            "prescriptions": f"{BASE}/mimic_iii/PRESCRIPTIONS.csv",
            "procedures_icd": f"{BASE}/mimic_iii/PROCEDURES_ICD.csv",
            "transfers": f"{BASE}/mimic_iii/TRANSFERS.csv",
        }

    elif ds == "eicu":
        ehr_dict = {
            "allergy": f"{BASE}/eicu/allergy.csv",
            "cost": f"{BASE}/eicu/cost.csv",
            "diagnosis": f"{BASE}/eicu/diagnosis.csv",
            "intakeoutput": f"{BASE}/eicu/intakeoutput.csv",
            "lab": f"{BASE}/eicu/lab.csv",
            "medication": f"{BASE}/eicu/medication.csv",
            "microlab": f"{BASE}/eicu/microlab.csv",
            "patient": f"{BASE}/eicu/patient.csv",
            "treatment": f"{BASE}/eicu/treatment.csv",
            "vitalperiodic": f"{BASE}/eicu/vitalperiodic.csv",
        }

    elif ds == "treqs":
        ehr_dict = {
            "DEMOGRAPHIC": f"{BASE}/treqs/treqs_db/DEMOGRAPHIC.csv",
            "DIAGNOSES": f"{BASE}/treqs/treqs_db/DIAGNOSES.csv",
            "LAB": f"{BASE}/treqs/treqs_db/LAB.csv",
            "PRESCRIPTIONS": f"{BASE}/treqs/treqs_db/PRESCRIPTIONS.csv",
            "PROCEDURES": f"{BASE}/treqs/treqs_db/PROCEDURES.csv",
        }

    else:
        raise KeyError(f"Unknown dataset '{dataset}'")

    key = _normalize_table_name(target_ehr, dataset=ds)
    if key not in ehr_dict:
        raise KeyError(
            f"Unknown table '{target_ehr}' for dataset={ds}. Available: {', '.join(sorted(ehr_dict.keys()))}"
        )

    data = pd.read_csv(ehr_dict[key])
    return data

def _strip_quotes(v: str):
    if not isinstance(v, str):
        return v
    v = v.strip()
    if len(v) >= 2 and ((v[0] == "'" and v[-1] == "'") or (v[0] == '"' and v[-1] == '"')):
        return v[1:-1].strip()
    return v

def _maybe_unwrap_singleton_list_literal(v: str) -> str:
    if not isinstance(v, str):
        return v
    s = v.strip()
    m = re.fullmatch(r"\[\s*(['\"])(.*?)\1\s*\]", s)
    if m:
        return m.group(2)
    m2 = re.fullmatch(r"\[\s*([0-9]+)\s*\]", s)
    if m2:
        return m2.group(1)
    return v


def _is_null_literal(v: str) -> bool:
    if not isinstance(v, str):
        return False
    return v.strip().lower() in {"none", "null", "nan", "na", "n/a"}


# next time try to remove it 
# def _first_non_null_exemplar(series: pd.Series):
#     try:
#         s = series.dropna()
#         if len(s) > 0:
#             return s.iloc[0]
#     except Exception:
#         pass
#     # fallback
#     try:
#         return series.iloc[0]
#     except Exception:
#         return None


def data_filter(data, argument):
    backup_data = data

    if not isinstance(argument, str):
        raise Exception("FilterDB argument must be a string.")

    argument = argument.replace("==", "=")
    commands = [c.strip() for c in argument.split("||") if c.strip()]

    for i in range(len(commands)):
        cmd = commands[i]
        column_name = None
        value = None

        try:
            if ">=" in cmd:
                command = cmd.split(">=")
                column_name = command[0].strip()
                value = _strip_quotes(command[1].strip())
                # exemplar = _first_non_null_exemplar(data[column_name])
                try:
                    # value = type(data)[column_name].tolist()[0](value) 
                    value = type(data[column_name].tolist()[0])(value)
                except:
                    pass
                data = data[data[column_name] >= value]

            elif "<=" in cmd:
                command = cmd.split("<=")
                column_name = command[0].strip()
                value = _strip_quotes(command[1].strip())
                # exemplar = _first_non_null_exemplar(data[column_name])
                try:
                    value = type(data[column_name].tolist()[0])(value) 
                except:
                    pass
                data = data[data[column_name] <= value]

            elif ">" in cmd:
                command = cmd.split(">")
                column_name = command[0].strip()
                value = _strip_quotes(command[1].strip())
                # exemplar = _first_non_null_exemplar(data[column_name])
                try:
                    value = type(data[column_name].tolist()[0])(value)
                except:      
                    pass              
                data = data[data[column_name] > value]

            elif "<" in cmd:
                command = cmd.split("<")
                column_name = command[0].strip()
                value = _strip_quotes(command[1].strip())
                # exemplar = _first_non_null_exemplar(data[column_name])
                try:
                    value = type(data[column_name].tolist()[0])(value)
                except:      
                    pass              
                data = data[data[column_name] < value]

            elif " in " in cmd:
                command = cmd.split(" in ")
                column_name = command[0].strip()
                value = command[1].strip()
                value_list = [s.strip() for s in value.strip("[]").split(",")]
                value_list = [s.strip("'").strip('"') for s in value_list if s.strip()]

                # exemplar = _first_non_null_exemplar(data[column_name])

                try:
                    value_list = list(map(type(data[column_name].tolist()[0]), value_list))
                except:      
                    pass              
                data = data[data[column_name].isin(value_list)]

            elif "max(" in cmd:
                command = cmd.split("max(")
                column_name = command[1].split(")")[0].strip()
                data = data[data[column_name] == data[column_name].max()]

            elif "min(" in cmd:
                command = cmd.split("min(")
                column_name = command[1].split(")")[0].strip()
                data = data[data[column_name] == data[column_name].min()]

            elif "=" in cmd:
                command = cmd.split("=", 1)
                column_name = command[0].strip()
                value = command[1].strip()

                value = _maybe_unwrap_singleton_list_literal(value)
                value = _strip_quotes(value)

                if _is_null_literal(value):
                    data = data[data[column_name].isna()]
                    continue

                # exemplar = _first_non_null_exemplar(backup_data[column_name])

                try:
                    exemplar = backup_data[column_name].tolist()[0]
                    value = type(exemplar)(value)
                    # value = type(exemplar)(value) if exemplar is not None else value
                except:
                    pass

                if pd.api.types.is_string_dtype(data[column_name]) or pd.api.types.is_object_dtype(data[column_name]):
                    v = str(value).lower()
                    data = data[data[column_name].astype(str).str.lower() == v]
                else:
                    data = data[data[column_name] == value]

            else:
                raise Exception(f"Unsupported filter command: {cmd}")

        except Exception:
            if (column_name is not None) and (column_name not in data.columns.tolist()):
                columns = ", ".join(data.columns.tolist())
                raise Exception(
                    "The filtering query {} is incorrect. Please modify the column name or use LoadDB to read another table. "
                    "The column names in the current DB are {}.".format(cmd, columns)
                )
            if (column_name is not None and column_name.strip() == "") or (value is not None and str(value).strip() == ""):
                raise Exception(
                    "The filtering query {} is incorrect. There is syntax error in the command. Please modify the condition or use LoadDB to read another table.".format(
                        cmd
                    )
                )
            raise

        if len(data) == 0 and column_name is not None:
            # series = backup_data[column_name]
            # try:
            #     sample_vals = series.dropna().astype(str).sample(n=min(2000, len(series.dropna())), random_state=0).tolist()
            # except Exception:
            #     sample_vals = [str(x) for x in series.dropna().tolist()[:2000]]

            column_values = list(set(backup_data[column_name].tolist()))
            if ("=" in cmd) and (not ">=" in cmd) and (not "<=" in cmd) and (value is not None) and (value not in column_values):
                levenshtein_dist = {}
                for cv in column_values:
                    levenshtein_dist[cv] = Levenshtein.distance(str(cv), str(value))
                levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
                examples = [x[0] for x in levenshtein_dist[:5]]
                examples = ", ".join([str(x) for x in examples])
                raise Exception(
                    "The filtering query {} is incorrect. There is no {} value in the column. Five example values in the column are {}. Please check if you get the correct {} value.".format(
                        cmd, value, examples, column_name
                    )
                )
            else:
                return data

    return data


def _clean_column_name(col: str) -> str:
    col = (col or "").strip()
    while col and (col[0] in ["[", "'", '"']):
        col = col[1:]
    while col and (col[-1] in ["]", "'", '"']):
        col = col[:-1]
    return col.strip()


def get_value(data, argument):
    column = None
    try:
        if not isinstance(argument, str) or not argument.strip():
            raise Exception("GetValue argument is empty.")
        if not hasattr(data, "columns"):
            raise Exception("GetValue expects a pandas DataFrame as the first argument.")

        arg = argument.strip()
        parts = [p.strip() for p in re.split(r"\s*,\s*", arg, maxsplit=1)]

        def _handle_missing_column(col: str, op: str | None):
            col_u = (col or "").upper().strip()

            if col_u == "DEATHTIME":
                if op == "count":
                    return 0
                if op == "list":
                    return []
                return ""

            if col_u == "HOSPITAL_EXPIRE_FLAG" and "DISCHARGE_LOCATION" in data.columns:
                series = data["DISCHARGE_LOCATION"].astype(str).str.lower()
                derived = series.str.contains("expired").astype(int)

                if op is None:
                    if len(data) == 0:
                        return ""

                    if len(data) == 1:
                        return str(int(derived.iloc[0]))
                    return [str(int(x)) for x in derived.tolist()[:200]]

                if len(data) == 0:
                    if op == "count":
                        return 0
                    if op == "list":
                        return []
                    return ""

                if op == "count":
                    return int(len(data))
                if op == "list":
                    return [str(int(x)) for x in derived.tolist()[:200]]

                if op in {"min", "max", "sum", "mean"}:
                    vals = pd.to_numeric(derived, errors="coerce").dropna()
                    if len(vals) == 0:
                        return ""
                    if op == "min":
                        return float(vals.min())
                    if op == "max":
                        return float(vals.max())
                    if op == "sum":
                        return float(vals.sum())
                    if op == "mean":
                        return float(vals.mean())
                return ""

            raise KeyError(col)

        if len(parts) == 1:
            column = _clean_column_name(parts[0])
            if column not in data.columns:
                return _handle_missing_column(column, op=None)

            if len(data) == 0:
                return ""

            if len(data) == 1:
                return str(data.iloc[0][column])

            vals = data[column].tolist()
            vals = [str(v) for v in vals]
            return vals[:200]

        column = _clean_column_name(parts[0])
        op = parts[1].lower().strip()

        if column not in data.columns:
            # print(data.columns.tolist())
            return _handle_missing_column(column, op=op)

        if len(data) == 0:
            if op == "count":
                return 0
            if op == "list":
                return []
            return ""

        if op == "count":
            return int(len(data))

        if op == "list":
            res_list = [str(i) for i in data[column].tolist()]
            return res_list[:200]

        if op in {"mean", "sum", "min", "max"}:
            series = pd.to_numeric(data[column], errors="coerce").dropna()
            if len(series) == 0:
                return ""
            if op == "mean":
                return float(series.mean())
            if op == "sum":
                return float(series.sum())
            if op == "min":
                return float(series.min())
            if op == "max":
                return float(series.max())

        raise Exception(
            "The operation {} contains syntax errors. Supported ops: mean, max, min, sum, list, count.".format(op)
        )

    except Exception:
        cols = ", ".join(data.columns.tolist()) if hasattr(data, "columns") else ""
        bad_col = column if column is not None else argument
        raise Exception(
            "The column name {} is incorrect OR the GetValue argument is malformed. "
            "Please check the column name/argument. The columns in this table include {}.".format(bad_col, cols)
        )

def sql_interpreter(command, dataset: str = "mimic_iii"):
    ds = _normalize_dataset(dataset)
    if ds == "mimic_iii":
        db_path = f"{BASE}/mimic_iii/mimic_iii.db"
    elif ds == "eicu":
        db_path = f"{BASE}/eicu/eicu.db"
    elif ds == "treqs":
        db_path = f"{BASE}/treqs/treqs_db/treqs_all.db"
    else:
        raise KeyError(f"Unknown dataset '{dataset}' for SQLInterpreter")

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(command)
    # rows = cur.fetchall()
    results = cur.execute(command).fetchall()

    # cols = [desc[0] for desc in cur.description] if cur.description else []
    # if cols:
    #     results = [dict(zip(cols, row)) for row in rows]
    # else:
    #     results = rows

    # cols = [desc[0] for desc in cur.description] if cur.description else []
    # if cols:
    #     norm_cols = []
    #     for c in cols:
    #         c = str(c)
    #         if "." in c:
    #             c = c.split(".")[-1]   # admissions.subject_id -> subject_id
    #         norm_cols.append(c.lower()) # SUBJECT_ID -> subject_id
    #     results = [dict(zip(norm_cols, row)) for row in rows]
    # else:
    #     results = rows

    con.close()
    return results


def date_calculator(argument, dataset: str = "mimic_iii"):
    ds = _normalize_dataset(dataset)
    if ds == "mimic_iii":
        db_path = f"{BASE}/mimic_iii/mimic_iii.db"
    elif ds == "eicu":
        db_path = f"{BASE}/eicu/eicu.db"
    elif ds == "treqs":
        db_path = f"{BASE}/treqs/treqs_db/treqs.db"
    else:
        raise KeyError(f"Unknown dataset '{dataset}' for Calendar")

    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        command = "select datetime(current_time, '{}')".format(argument)
        results = cur.execute(command).fetchall()[0][0]
        con.close()
    except Exception:
        raise Exception(
            "The date calculator {} is incorrect. Please check the syntax and make necessary changes. "
            "For the current date and time, please call Calendar('0 year').".format(argument)
        )
    return results


if __name__ == "__main__":
    pass
