import json
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
        "hadm_id","subject_id","name","marital_status","age",
        "date_of_birth","gender","language","religion","admission_type",
        "days_of_hospital_stay","insurance","ethnicity","death_status",
        "admission_location","discharge_location","primary_disease",
        "date_of_death","year_of_birth","year_of_death",
        "admission_time","discharge_time","admission_year"
    ],

    "DIAGNOSES":[
        "subject_id","hadm_id","icd9_code","short_title","long_title"
    ],

    "PROCEDURES":[
        "subject_id","hadm_id","icd9_code","short_title","long_title"
    ],

    "PRESCRIPTIONS":[
        "subject_id","hadm_id","icustay_id","drug_type","drug_name",
        "drug_code","drug_route","drug_dose"
    ],

    "LAB":[
        "subject_id","hadm_id","itemid","charttime",
        "abnormal_flag","lab_value","label","fluid","category"
    ]
}

table_usage = Counter()
column_usage = Counter()
errors = []

with open(DATA_PATH) as f:
    for line_id,line in enumerate(f):

        row = json.loads(line)

        fmt = row.get("format",{})

        tables = fmt.get("table",[])
        conds = fmt.get("cond",[])

        for t in tables:
            if t not in TABLE_MAP:
                errors.append(("unknown_table",line_id,t))
            else:
                table_usage[TABLE_MAP[t]] += 1

        for c in conds:

            if len(c) != 4:
                continue

            table_id,col_id,op,val = c

            if table_id not in TABLE_MAP:
                errors.append(("bad_table_in_cond",line_id,table_id))
                continue

            table_name = TABLE_MAP[table_id]

            cols = COLUMN_MAP[table_name]

            if col_id >= len(cols):
                errors.append(("bad_column",line_id,table_name,col_id))
            else:
                column_usage[(table_name,cols[col_id])] += 1


print("\nTABLE USAGE")
for k,v in table_usage.items():
    print(k,v)

print("\nCOLUMN USAGE")
for k,v in column_usage.most_common(20):
    print(k,v)

print("\nERRORS")
for e in errors[:20]:
    print(e)

print("\nTOTAL ERRORS:",len(errors))
