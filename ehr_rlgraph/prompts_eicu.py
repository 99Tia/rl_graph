from __future__ import annotations

CodeHeader = """from tools import tabtools, calculator
Calculate = calculator.WolframAlphaCalculator
LoadDB = tabtools.db_loader
FilterDB = tabtools.data_filter
GetValue = tabtools.get_value
SQLInterpreter = tabtools.sql_interpreter
Calendar = tabtools.date_calculator
"""

RetrKnowledge = """Read the following eICU table descriptions, then generate brief background knowledge (context) to help answer the question. Focus on:
- which table(s) likely contain the needed info,
- which column(s) are relevant (use exact column names),
- how to link tables using keys (patientunitstayid, patienthealthsystemstayid, uniquepid),
- what filtering/aggregation is needed (min/max/sum/count/distinct),
- and whether SQLInterpreter is required (for LIKE, joins, last/first stay).

General reminders:
(1) eICU is a relational database. Many tables join on patientunitstayid (ICU stay).
(2) Different units may miss interfaces → some stays have no rows in some tables.
(3) If you need partial string match (contains), use SQLInterpreter with lower(col) like '%...%'.
(4) If you need "last/first encounter" for a patient, use SQL ORDER BY ... LIMIT 1 (do NOT use FilterDB max()/min()).

Core keys:
- patientunitstayid: ICU stay id (main join key for most tables)
- patienthealthsystemstayid: hospital stay id
- uniquepid: patient id across stays

Tables and columns (eICU):
(1) allergy: allergyid, patientunitstayid, drugname, allergyname, allergytime
(2) cost: costid, uniquepid, patienthealthsystemstayid, eventtype, eventid, chargetime, cost
(3) diagnosis: diagnosisid, patientunitstayid, icd9code, diagnosisname, diagnosistime
(4) intakeoutput: intakeoutputid, patientunitstayid, cellpath, celllabel, cellvaluenumeric, intakeoutputtime
(5) lab: labid, patientunitstayid, labname, labresult, labresulttime
(6) medication: medicationid, patientunitstayid, drugname, dosage, routeadmin, drugstarttime, drugstoptime
(7) microlab: microlabid, patientunitstayid, culturesite, organism, culturetakentime
(8) patient: patientunitstayid, patienthealthsystemstayid, gender, age, ethnicity, hospitalid, wardid,
            admissionheight, hospitaladmitsource, hospitaldischargestatus, admissionweight, dischargeweight,
            uniquepid, hospitaladmittime, unitadmittime, unitdischargetime, hospitaldischargetime
(9) treatment: treatmentid, patientunitstayid, treatmentname, treatmenttime
(10) vitalperiodic: vitalperiodicid, patientunitstayid, temperature, sao2, heartrate, respiration,
                   systemicsystolic, systemicdiastolic, systemicmean, observationtime

If unsure about schema, check real columns with:
- SQLInterpreter("PRAGMA table_info(patient)")
- SQLInterpreter("PRAGMA table_info(medication)")
- SQLInterpreter("PRAGMA table_info(vitalperiodic)")
- SQLInterpreter("PRAGMA table_info(cost)")

Question: {question}
Knowledge:
"""

EHRAgent_Message_Prompt = """Assume you have knowledge of several eICU tables.

Write python code to solve the question using ONLY the functions below:

(1) Calculate(FORMULA): returns the computed result (use ONLY for simple arithmetic, NOT datetime).
(2) LoadDB(DBNAME): loads a table and returns it.
(3) FilterDB(DATABASE, CONDITIONS): filters DATABASE by CONDITIONS (a string with multiple conditions separated by "||").
(4) GetValue(DATABASE, ARGUMENT): returns values from DATABASE.
    ARGUMENT examples: "patientunitstayid" or "cost, sum" or "uniquepid, list"
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
0) LoadDB takes EXACTLY ONE table name string.
   BAD:  LoadDB("allergy, cost, diagnosis")
   GOOD: LoadDB("allergy")

1) NEVER write list-literal with '=':
   BAD:  patientunitstayid=[123,456]
   GOOD: patientunitstayid in [123,456]

2) NO parentheses in FilterDB:
   BAD:  (A=1||B=2)||C=3
   GOOD: A=1||B=2||C=3
   If you need OR logic, use SQLInterpreter.

3) FilterDB does NOT support LIKE/regex or contains matching.
   If you need partial string match, use SQLInterpreter with:
   lower(column) like '%substring%'

4) Per-patient "last/first encounter" selection: use SQLInterpreter, NOT FilterDB max()/min().
   Example (last ICU stay for a patient uniquepid):
   rows = SQLInterpreter("select patientunitstayid from patient where uniquepid='XXX' order by unitadmittime desc limit 1")
   stay_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0].get('patientunitstayid') if rows else None)

5) Do NOT use Calculate() for datetime differences. Use SQLInterpreter datetime logic.

6) IMPORTANT execution environment:
   - Each python tool call runs in a fresh process.
   - Therefore, define ALL needed variables inside ONE tool call.

Other tool rules:
- Do NOT use NULL/null in FilterDB (unsupported). If needed, use SQLInterpreter "is null".
- SQLInterpreter can only query real DB tables, not Python variables.
- GetValue supports ONLY one column at a time (no multi-column requests).

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
(1) Calculate(FORMULA): arithmetic only (NOT datetime)
(2) LoadDB(DBNAME): one table name only
(3) FilterDB(DATABASE, CONDITIONS) where CONDITIONS uses "||" between conditions.
    Supported FilterDB grammar:
    - COLUMN=VALUE, COLUMN>=VALUE, COLUMN<=VALUE, COLUMN>VALUE, COLUMN<VALUE
    - COLUMN in [v1, v2, ...]
    - max(COLUMN), min(COLUMN)
    - NO parentheses
(4) GetValue(DATABASE, ARGUMENT) one column only
(5) SQLInterpreter(SQL)
(6) Calendar(DURATION)

The code is:
{code}

The execution result / error is:
{error_info}

Please point out the single most likely reason for the error (brief and specific).
Check for:
- LoadDB called with multiple table names (comma-separated)
- FilterDB using list literal with '=' instead of 'in'
- parentheses in FilterDB
- per-patient last/first using FilterDB max()/min() (wrong; need SQL ORDER BY LIMIT 1)
- invented columns
- SQLInterpreter querying python variables instead of DB tables
- Calculate used for datetime subtraction
- multi-tool-call variable usage (NameError due to fresh env)
"""

EHRAgent_4Shots_Knowledge = """Question: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205 on their current hospital encounter?
Knowledge:
- Use SQL to get the most recent ICU stay (patientunitstayid) for uniquepid=035-2205 ordered by unitadmittime desc.
- Then check medication for that patientunitstayid and exact drugname match.
Solution:
rows = SQLInterpreter("select patientunitstayid from patient where uniquepid='035-2205' order by unitadmittime desc limit 1")
stay_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0].get('patientunitstayid') if rows else None)

med_db = LoadDB('medication')
med_f = FilterDB(med_db, "patientunitstayid={}||drugname=fluticasone-salmeterol 250-50 mcg/dose in aepb".format(stay_id))
answer = 1 if len(med_f) > 0 else 0

Question: in the last hospital encounter, when was patient 031-22988's first microbiology test time?
Knowledge:
- Use SQL to get the most recent stay_id for uniquepid, then use microlab and take min(culturetakentime) (SQL is safer).
Solution:
rows = SQLInterpreter("select patientunitstayid from patient where uniquepid='031-22988' order by unitadmittime desc limit 1")
stay_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0].get('patientunitstayid') if rows else None)

rows2 = SQLInterpreter("select min(culturetakentime) from microlab where patientunitstayid={}".format(stay_id))
answer = rows2[0][0] if rows2 and isinstance(rows2[0], (list, tuple)) else (rows2[0].get('min(culturetakentime)') if rows2 else None)

Question: what is the minimum hospital cost for a drug with a name called albumin 5% since 6 years ago?
Knowledge:
- Use SQL join: medication (drugname) -> patient (patienthealthsystemstayid) -> cost (patienthealthsystemstayid), filter chargetime >= Calendar('-6 year'), take min(cost).
Solution:
date = Calendar('-6 year')
q = (
  "select min(c.cost) "
  "from cost c "
  "join patient p on p.patienthealthsystemstayid = c.patienthealthsystemstayid "
  "join medication m on m.patientunitstayid = p.patientunitstayid "
  "where m.drugname = 'albumin 5%' and c.chargetime >= '{}'".format(date)
)
rows = SQLInterpreter(q)
answer = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0].get('min(c.cost)') if rows else None)

Question: what is the intake method of albumin 5%?
Knowledge:
- Intake method is medication.routeadmin. Filter medication by drugname and return routeadmin.
Solution:
med_db = LoadDB('medication')
med_f = FilterDB(med_db, "drugname=albumin 5%")
answer = GetValue(med_f, "routeadmin")
"""