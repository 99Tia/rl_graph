CodeHeader = """from tools import tabtools, calculator
Calculate = calculator.WolframAlphaCalculator
LoadDB = tabtools.db_loader
FilterDB = tabtools.data_filter
GetValue = tabtools.get_value
SQLInterpreter = tabtools.sql_interpreter
Calendar = tabtools.date_calculator
"""

RetrKnowledge = """Read the following data descriptions, then generate brief background knowledge (as context) that could help answer the question. Focus on:
- which table(s) likely contain the needed info,
- which column(s) are relevant (use exact column names),
- how to link tables using IDs (SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, ICD9_CODE),
- and what filtering/aggregation is needed.

General reminders:
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID is a patient, HADM_ID is a hospital admission, and ICUSTAY_ID is an ICU stay.
(2) Events tables store measurements and time-stamped records (e.g., chartevents, labevents, outputevents, inputevents_cv).
(3) Dictionary tables (d_*) map codes/ITEMID to names; join on ITEMID or ICD9_CODE.
(4) Core stay tables: admissions, patients, icustays, transfers; other tables store care data.

For different tables, they contain the following columns:
(1) admissions: ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, MARITAL_STATUS, ETHNICITY, AGE
(2) chartevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(3) cost: ROW_ID, SUBJECT_ID, HADM_ID, EVENT_TYPE, EVENT_ID, CHARGETIME, COST
(4) d_icd_diagnoses: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(5) d_icd_procedures: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(6) d_items: ROW_ID, ITEMID, LABEL, LINKSTO
(7) d_labitems: ROW_ID, ITEMID, LABEL
(8) diagnoses_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(9) icustays: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, FIRST_CAREUNIT, LAST_CAREUNIT, FIRST_WARDID, LAST_WARDID, INTIME, OUTTIME
(10) inputevents_cv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, AMOUNT
(11) labevents: ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(12) microbiologyevents: ROW_ID, SUBJECT_ID, HADM_ID, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME
(13) outputevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
(14) patients: ROW_ID, SUBJECT_ID, GENDER, DOB, DOD
(15) prescriptions: ROW_ID, SUBJECT_ID, HADM_ID, STARTDATE, ENDDATE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE
(16) procedures_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(17) transfers: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, EVENTTYPE, CAREUNIT, WARDID, INTIME, OUTTIME

Column sanity reminders (based on our MIMIC-III setup):
- admissions does NOT have DEATHTIME or HOSPITAL_EXPIRE_FLAG.
  If asked whether patient died in hospital:
  - Use DISCHARGE_LOCATION contains "EXPIRED" as proxy, OR
  - Use patients table if it contains death information (e.g., DOD).
- icustays missing OUTTIME may appear as NaN; in FilterDB you can use OUTTIME=None (it matches missing).
- inputevents_cv uses AMOUNT.
- outputevents uses VALUE.

If unsure about a table schema, check real columns with:
- SQLInterpreter("PRAGMA table_info(admissions)")
- SQLInterpreter("PRAGMA table_info(icustays)")
- SQLInterpreter("PRAGMA table_info(outputevents)")
- SQLInterpreter("PRAGMA table_info(inputevents_cv)")

IMPORTANT retrieval reminder:
- If you need "last / first admission" or "per-subject max/min time", prefer SQL (ORDER BY ... LIMIT 1).
  FilterDB max(ADMITTIME) is GLOBAL max across whole table, not grouped per patient.

Question: {question}
Knowledge:
"""

EHRAgent_Message_Prompt = """Assume you have knowledge of several tables:
(1) Tables are linked by identifiers with suffix 'ID' (SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, ICD9_CODE).
(2) Events tables store measurements; dictionary tables (d_*) map codes/ITEMID to names.

Write python code to solve the question using ONLY the functions below:

(1) Calculate(FORMULA): returns the computed result (use ONLY for simple arithmetic, NOT datetime).
(2) LoadDB(DBNAME): loads a table and returns it.
(3) FilterDB(DATABASE, CONDITIONS): filters DATABASE by CONDITIONS (a string with multiple conditions separated by "||").
(4) GetValue(DATABASE, ARGUMENT): returns values from DATABASE.
    ARGUMENT examples: "HADM_ID" or "COST, sum" or "HADM_ID, list"
    IMPORTANT: do NOT include quotes inside ARGUMENT.
(5) SQLInterpreter(SQL): executes SQL and returns the result (list of rows).
(6) Calendar(DURATION): returns a datetime string after duration (e.g., "-1 year").

Use variable `answer` for the final answer.

IMPORTANT (Tool calling format):
- You MUST call the python tool to execute the code.
- Do NOT write code as plain text.
- Do NOT wrap code in markdown fences (no ```).
- Return ONLY a python tool call with {{"cell": "..."}}.
- The code MUST assign the final result to variable `answer`.

FilterDB grammar (MUST follow exactly):
- Equality:      COLUMN=VALUE
- Inequality:    COLUMN>=VALUE, COLUMN<=VALUE, COLUMN>VALUE, COLUMN<VALUE
- Membership:    COLUMN in [v1, v2, v3]
- Aggregation:   max(COLUMN) or min(COLUMN)
- Multiple cond: join with "||"
- NO parentheses are allowed in CONDITIONS.

CRITICAL RULES (these errors cause failures):
1) NEVER write list-literal with '=':
   BAD:  ICD9_CODE=['80702']
   GOOD: ICD9_CODE=80702
   GOOD: ICD9_CODE in [80702, 80707]

2) NO parentheses in FilterDB:
   BAD:  (A=1||B=2)||C=3
   GOOD: A=1||B=2||C=3
   If you need OR logic, use SQLInterpreter.

3) Per-patient "last/first" selection: use SQLInterpreter, NOT FilterDB max()/min().
   Example (last admission):
   rows = SQLInterpreter("select hadm_id from admissions where subject_id=2238 order by admittime desc limit 1")
   hadm_id = rows[0][0]  (if SQL returns tuples)
   OR hadm_id = rows[0]['hadm_id'] (if SQL returns dict rows)
   If you are unsure which format you got, print/inspect `rows` and adapt.

4) Do NOT invent columns. If unsure, check schema with:
   SQLInterpreter("PRAGMA table_info(admissions)")

5) Do NOT use Calculate() for datetime differences. Use SQLInterpreter for time arithmetic.

6) Use correct amount columns:
   - inputevents_cv: AMOUNT
   - outputevents: VALUE

7) FilterDB does NOT support LIKE/regex. Use SQLInterpreter for LIKE or complex joins.

8) IMPORTANT execution environment:
   - Each python tool call runs in a fresh process.
   - Therefore, define ALL needed variables inside ONE tool call.
   - Do not rely on variables from a previous tool call.

Other tool rules:
- FilterDB does NOT support NULL keyword; for missing ICU OUTTIME use OUTTIME=None.
- SQLInterpreter can only query real DB tables, not Python variables.
- GetValue supports ONLY one column at a time (no multi-column requests).
- Specimen name comes from microbiologyevents.SPEC_TYPE_DESC.
- prescriptions has DRUG (not ITEMID). Don’t join to d_items for drug names.

Here are some examples:
{examples}
(END OF EXAMPLES)

Knowledge:
{knowledge}

Question: {question}
Solution: """

DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
    "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
    "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
    "NEVER": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
}

CodeDebugger = """Given a question:
{question}

The user has written code using these functions:
(1) Calculate(FORMULA)
(2) LoadDB(DBNAME)
(3) FilterDB(DATABASE, CONDITIONS) where CONDITIONS uses "||" between conditions.
    Supported FilterDB grammar:
    - COLUMN=VALUE, COLUMN>=VALUE, COLUMN<=VALUE, COLUMN>VALUE, COLUMN<VALUE
    - COLUMN in [v1, v2, ...]
    - max(COLUMN), min(COLUMN)
    - NO parentheses
(4) GetValue(DATABASE, ARGUMENT) where ARGUMENT can be "COLUMN" or "COLUMN, op" (e.g., "COST, sum"). Do not include quotes in ARGUMENT.
(5) SQLInterpreter(SQL)
(6) Calendar(DURATION)

The code is:
{code}

The execution result / error is:
{error_info}

Please point out the single most likely reason for the error (brief and specific).
Check for:
- FilterDB using list literal with '=' (ICD9_CODE=['80702']) instead of 'in'
- parentheses in FilterDB (unsupported)
- per-patient last/first using FilterDB max()/min() (wrong; need SQL ORDER BY LIMIT 1)
- invented columns (wrong column names)
- wrong table name (inputevents vs inputevents_cv)
- wrong amount column name (VOLUME vs AMOUNT; outputevents uses VALUE)
- Calculate used for datetime subtraction
- SQLInterpreter querying python variables instead of DB tables
- multi-tool-call variable usage (NameError due to fresh env)
"""

EHRAgent_4Shots_Knowledge = """Question: What is the maximum total hospital cost that involves a diagnosis named comp-oth vasc dev/graft since 1 year ago?
Knowledge:
- Find ICD9_CODE from d_icd_diagnoses using SHORT_TITLE.
- Use diagnoses_icd to find HADM_ID for that ICD9_CODE, filtered by CHARTTIME >= Calendar('-1 year').
- For those HADM_ID, sum cost.COST and take the maximum.
Solution:
date = Calendar('-1 year')

diag = LoadDB('d_icd_diagnoses')
diag_f = FilterDB(diag, 'SHORT_TITLE=comp-oth vasc dev/graft')
icd = GetValue(diag_f, 'ICD9_CODE')

dx = LoadDB('diagnoses_icd')
dx_f = FilterDB(dx, 'ICD9_CODE={}||CHARTTIME>={}'.format(icd, date))
hadm_ids = GetValue(dx_f, 'HADM_ID, list')

cost = LoadDB('cost')
max_cost = 0.0
for hadm in hadm_ids:
    c_f = FilterDB(cost, 'HADM_ID={}'.format(hadm))
    c_sum = GetValue(c_f, 'COST, sum')
    try:
        c_val = float(c_sum)
    except Exception:
        c_val = float(Calculate(str(c_sum)))
    if c_val > max_cost:
        max_cost = c_val

answer = max_cost

Question: had any tpn w/lipids been given to patient 2238 in their last hospital visit?
Knowledge:
- Get last HADM_ID for SUBJECT_ID=2238 from admissions using SQL ORDER BY ADMITTIME DESC LIMIT 1.
- Map LABEL 'tpn w/lipids' -> ITEMID via d_items.
- Find ICUSTAY_ID via icustays for that HADM_ID (if needed).
- Check inputevents_cv for any row with that HADM_ID, ICUSTAY_ID, ITEMID.
Solution:
# Use SQL for "last admission" per patient (avoid FilterDB max(ADMITTIME) which is global)
rows = SQLInterpreter("select hadm_id from admissions where subject_id=2238 order by admittime desc limit 1")
hadm_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0].get('hadm_id') if rows else None)

items = LoadDB('d_items')
items_f = FilterDB(items, 'LABEL=tpn w/lipids')
item_id = GetValue(items_f, 'ITEMID')

icu = LoadDB('icustays')
icu_f = FilterDB(icu, 'HADM_ID={}'.format(hadm_id))
icustay_id = GetValue(icu_f, 'ICUSTAY_ID')

try:
    inp = LoadDB('inputevents')
except Exception:
    inp = LoadDB('inputevents_cv')

inp_f = FilterDB(inp, 'HADM_ID={}||ICUSTAY_ID={}||ITEMID={}'.format(hadm_id, icustay_id, item_id))
answer = 1 if len(inp_f) > 0 else 0

Question: what was the name of the procedure that was given two or more times to patient 58730?
Knowledge:
- Get ICD9_CODE counts from procedures_icd for admissions of SUBJECT_ID=58730.
- Map ICD9_CODE to SHORT_TITLE using d_icd_procedures.
Solution:
answer = SQLInterpreter(
    "select d_icd_procedures.short_title "
    "from d_icd_procedures "
    "where d_icd_procedures.icd9_code in ("
    "  select t1.icd9_code from ("
    "    select procedures_icd.icd9_code, count(procedures_icd.charttime) as c1 "
    "    from procedures_icd "
    "    where procedures_icd.hadm_id in ("
    "      select admissions.hadm_id from admissions where admissions.subject_id = 58730"
    "    ) "
    "    group by procedures_icd.icd9_code"
    "  ) as t1 where t1.c1 >= 2"
    ")"
)

Question: what is the intake method of clobetasol propionate 0.05% ointment?
Knowledge:
- Intake method is prescriptions.ROUTE. Filter prescriptions by DRUG then return ROUTE.
Solution:
rx = LoadDB('prescriptions')
rx_f = FilterDB(rx, 'DRUG=clobetasol propionate 0.05% ointment')
answer = GetValue(rx_f, 'ROUTE')
"""