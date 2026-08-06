from __future__ import annotations

CodeHeader = """from tools import tabtools, calculator
Calculate = calculator.WolframAlphaCalculator
LoadDB = tabtools.db_loader
FilterDB = tabtools.data_filter
GetValue = tabtools.get_value
SQLInterpreter = tabtools.sql_interpreter
Calendar = tabtools.date_calculator
"""

RetrKnowledge = """Read the following TREQS table descriptions, then generate SHORT, STRICT, and benchmark-oriented background knowledge.

Focus ONLY on:
- which table(s) contain the needed info,
- which exact column(s) are relevant,
- how to link tables using SUBJECT_ID / HADM_ID,
- whether the answer should be a scalar or list,
- and whether exact DB values should be inspected first.

STRICT rules:
(1) Do not invent hidden meanings for English phrases.
(2) If a wording may not exactly match DB values, explicitly say:
    "inspect exact DB values first with SQLInterpreter(select distinct ...)".
(3) If the question asks for ONE NUMBER, the answer should be ONE scalar.
(4) If the question says "primary disease", prefer DEMOGRAPHIC.DIAGNOSIS first.
(5) If the question says "how many patients", usually use COUNT(DISTINCT SUBJECT_ID).
(6) If the question is about admissions, hospital admissions, or admission rows, think carefully whether HADM_ID is the right counting unit.
(7) Prefer exact equality before LIKE.
(8) Use SQLInterpreter for exact DB-value inspection when uncertain.

Schema:
- DEMOGRAPHIC: SUBJECT_ID, HADM_ID, NAME, MARITAL_STATUS, AGE, DOB, GENDER, LANGUAGE, RELIGION, ADMISSION_TYPE, DAYS_STAY, INSURANCE, ETHNICITY, EXPIRE_FLAG, ADMISSION_LOCATION, DISCHARGE_LOCATION, DIAGNOSIS, DOD, DOB_YEAR, DOD_YEAR, ADMITTIME, DISCHTIME, ADMITYEAR
- DIAGNOSES: SUBJECT_ID, HADM_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
- LAB: SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, FLAG, VALUE_UNIT, LABEL, FLUID, CATEGORY
- PRESCRIPTIONS: SUBJECT_ID, HADM_ID, ICUSTAY_ID, DRUG_TYPE, DRUG, FORMULARY_DRUG_CD, ROUTE, DRUG_DOSE
- PROCEDURES: SUBJECT_ID, HADM_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE

Important reminders:
- Use SUBJECT_ID and HADM_ID to link tables.
- DEMOGRAPHIC.DIAGNOSIS is often best for "primary disease".
- DIAGNOSES / PROCEDURES do not have order columns.
- LAB has no numeric lab-result column in this setup.
- If uncertain about an exact value, inspect first:
  - SQLInterpreter("select distinct ADMISSION_TYPE from DEMOGRAPHIC order by 1")
  - SQLInterpreter("select distinct ADMISSION_LOCATION from DEMOGRAPHIC order by 1")
  - SQLInterpreter("select distinct DISCHARGE_LOCATION from DEMOGRAPHIC order by 1")
  - SQLInterpreter("select distinct LABEL from LAB order by 1")
  - SQLInterpreter("select distinct SHORT_TITLE from DIAGNOSES order by 1")
  - SQLInterpreter("select distinct LONG_TITLE from DIAGNOSES order by 1")
  - SQLInterpreter("select distinct SHORT_TITLE from PROCEDURES order by 1")
  - SQLInterpreter("select distinct LONG_TITLE from PROCEDURES order by 1")

Question: {question}
Knowledge:
"""

EHRAgent_Message_Prompt = """Assume you have knowledge of several TREQS tables:
(1) Tables are linked mainly by SUBJECT_ID and HADM_ID.
(2) DEMOGRAPHIC contains admission/discharge and demographic information.
(3) DIAGNOSES and PROCEDURES contain ICD9_CODE plus titles.
(4) PRESCRIPTIONS contains DRUG, ROUTE, and DRUG_DOSE.
(5) LAB contains ITEMID, LABEL, FLUID, CATEGORY, FLAG, VALUE_UNIT, and CHARTTIME.

Write python code to solve the question using ONLY the functions below:

(1) Calculate(FORMULA): returns the computed result (use ONLY for simple arithmetic, NOT datetime).
(2) LoadDB(DBNAME): loads a table and returns it.
(3) FilterDB(DATABASE, CONDITIONS): filters DATABASE by CONDITIONS (a string with multiple conditions separated by "||").
(4) GetValue(DATABASE, ARGUMENT): returns values from DATABASE.
(5) SQLInterpreter(SQL): executes SQL and returns the result.
(6) Calendar(DURATION): returns a datetime string after duration (e.g., "-1 year").

Use variable `answer` for the final answer.

IMPORTANT (Tool calling format):
- You MUST call the python tool.
- Return ONLY a python tool call with {{"cell": "..."}}.
- The code MUST assign the final result to variable `answer`.
- Do NOT write explanation text outside the tool call.

FilterDB grammar:
- Equality:      COLUMN=VALUE
- Inequality:    COLUMN>=VALUE, COLUMN<=VALUE, COLUMN>VALUE, COLUMN<VALUE
- Membership:    COLUMN in [v1, v2, v3]
- Aggregation:   max(COLUMN) or min(COLUMN)
- Multiple cond: join with "||"
- NO parentheses in CONDITIONS.

CRITICAL BENCHMARK RULES:
1) If the question asks for ONE NUMBER, return ONE SCALAR only.
   - Do NOT return a list.
   - Do NOT return all rows.
   - Do NOT return multiple candidate values.

2) If the question says "how many patients", usually use:
   COUNT(DISTINCT SUBJECT_ID)

3) If the question says "admissions", "hospital admissions", or clearly refers to admission rows,
   think carefully whether HADM_ID or row count is the right unit.
   Do NOT automatically count SUBJECT_ID.

4) If the question says "primary disease", prefer DEMOGRAPHIC.DIAGNOSIS first.
   Use DIAGNOSES only if the question explicitly asks for diagnosis code / diagnosis short title / diagnosis long title.

5) Prefer exact equality before LIKE:
   - codes -> always exact equality
   - language / insurance / ethnicity / route / drug code / itemid / title-like values:
     use exact equality first if the question gives a specific value
   - use LIKE only when wording is clearly partial or exact DB value is unknown

6) If an English phrase may not exactly match DB values, inspect DB values first using SQLInterpreter(select distinct ...).
   Examples:
   - inpatient
   - transfer within facility
   - diagnosis / procedure title phrases
   - lab label phrases
   - admission/discharge location phrases

7) Do NOT invent hidden mappings such as:
   - inpatient = ADMISSION_TYPE 'INPATIENT'
   - within the facility = value contains 'transfer'
   unless you verify exact DB values first.

8) For disease phrases:
   - if the question explicitly says diagnosis icd9 code -> use ICD9_CODE
   - if it explicitly says diagnosis short title / long title -> use that field
   - if it says primary disease -> prefer DEMOGRAPHIC.DIAGNOSIS
   - do not automatically map disease name to ICD9 prefix unless the benchmark wording clearly indicates code grouping

9) For lab questions:
   - LABEL may not be enough by itself
   - if wording may be ambiguous, inspect LABEL / FLUID / CATEGORY values first
   - think carefully whether benchmark wants patients, admissions, or exact lab-value grouping

10) For list-vs-scalar interpretation:
   - "how many", "what is the number", "calculate the total number" -> scalar
   - "average", "minimum", "maximum", "sum" -> scalar
   - "for how long" in this benchmark often expects one scalar, not a whole list
   - return a list only when the question clearly asks for items/values/records

11) SQLInterpreter is usually safer than FilterDB when:
   - exact DB values are uncertain
   - joins are needed
   - patient vs admission counting matters
   - exact equality vs LIKE matters
   - one scalar answer is required

12) Do NOT invent columns.
    If unsure, inspect schema with:
    SQLInterpreter("PRAGMA table_info(DEMOGRAPHIC)")
    SQLInterpreter("PRAGMA table_info(DIAGNOSES)")
    SQLInterpreter("PRAGMA table_info(LAB)")
    SQLInterpreter("PRAGMA table_info(PRESCRIPTIONS)")
    SQLInterpreter("PRAGMA table_info(PROCEDURES)")

13) DIAGNOSES and PROCEDURES do not have diagnosis/procedure ordering columns.
    Do not invent primary/first ordering there.

14) Each python tool call runs in a fresh process.
    Define everything inside one tool call.

15) GetValue supports one column only.
    If you need multi-column output, prefer SQLInterpreter.

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

The user has written code using:
(1) Calculate(FORMULA)
(2) LoadDB(DBNAME)
(3) FilterDB(DATABASE, CONDITIONS)
(4) GetValue(DATABASE, ARGUMENT)
(5) SQLInterpreter(SQL)
(6) Calendar(DURATION)

The code is:
{code}

The execution result / error is:
{error_info}

Please point out the single most likely reason for the error or wrong answer.

Check for:
- wrong counting unit: SUBJECT_ID vs HADM_ID vs row count
- returning a list when benchmark expects one scalar
- using LIKE where exact equality is needed
- assuming a DB value without inspecting distinct values first
- using DIAGNOSES when question really means DEMOGRAPHIC.DIAGNOSIS
- wrong join keys
- wrong table choice
- invented columns
- assuming diagnosis/procedure order that is not in schema
- scalar/list mismatch
- Python syntax bug
"""

EHRAgent_4Shots_Knowledge = """Question: how many patients were born before the year 2104 and have medicare insurance?
Knowledge:
- Use DEMOGRAPHIC.
- The question asks for number of patients, so count DISTINCT SUBJECT_ID.
- DOB_YEAR and INSURANCE are enough.
- Use exact equality for INSURANCE.
Solution:
rows = SQLInterpreter("select count(distinct SUBJECT_ID) from DEMOGRAPHIC where DOB_YEAR < 2104 and INSURANCE = 'Medicare'")
answer = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0].get('count(distinct SUBJECT_ID)') if rows else None)


Question: how many patients speak cape language and are under 71 years of age?
Knowledge:
- Use DEMOGRAPHIC.
- The question asks for number of patients, so count DISTINCT SUBJECT_ID.
- LANGUAGE and AGE are enough.
- Use exact equality for LANGUAGE here.
Solution:
rows = SQLInterpreter("select count(distinct SUBJECT_ID) from DEMOGRAPHIC where lower(LANGUAGE) = 'cape' and AGE < 71")
answer = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0] if rows else None)


Question: what is the average age of patients whose primary disease is bowel obstruction and who are aged 71 years or older?
Knowledge:
- "primary disease" should prefer DEMOGRAPHIC.DIAGNOSIS.
- Use DEMOGRAPHIC only.
- The answer is one scalar average value.
Solution:
rows = SQLInterpreter('''
select avg(AGE)
from DEMOGRAPHIC
where AGE >= 71 and lower(DIAGNOSIS) like '%bowel obstruction%'
''')
answer = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else (rows[0].get('avg(AGE)') if rows else None)


Question: what is the drug name used by the patient stephanie suchan?
Knowledge:
- Use DEMOGRAPHIC to get SUBJECT_ID from NAME.
- Use PRESCRIPTIONS to get DRUG.
- Return the relevant values only.
Solution:
rows = SQLInterpreter("select subject_id from DEMOGRAPHIC where lower(NAME) = 'stephanie suchan'")
subject_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else None
if subject_id is not None:
    drugs = SQLInterpreter(f"select distinct DRUG from PRESCRIPTIONS where SUBJECT_ID={subject_id}")
    answer = [row[0] for row in drugs] if drugs else []
else:
    answer = []
"""