# 🧠 EHR-RLGraph  
### Graph Memory + RL Retrieval for EHR Question Answering

<p align="center">
  <img src="https://img.shields.io/badge/Framework-LLM%20Agent-blue" />
  <img src="https://img.shields.io/badge/Method-RL%20Retrieval-green" />
  <img src="https://img.shields.io/badge/Data-EHR-red" />
</p>

---


## 📝 About This README

This README is intended to guide how to run the project. It first provides a brief introduction to the work, followed by step-by-step instructions for setting up and running the project on the server.


## 📌 Overview

**EHR-RLGraph** is a large language model (LLM)–based agent for answering **complex clinical questions over structured Electronic Health Records (EHRs)**.

Real-world EHR reasoning requires:
- multi-step reasoning across tables  
- correct schema understanding  
- robust execution of queries  

However, existing methods rely on **static demonstration retrieval**, which often fails for complex queries.

### 🚀 Our Solution

We introduce a framework that combines:

- 🧩 **Graph-based memory** of execution-verified demonstrations  
- 🎯 **Reinforcement learning (RL)–based retriever**  
- 🔁 **Interactive code generation with execution feedback**  

The agent dynamically retrieves **structurally relevant demonstrations**, generates Python programs, executes them, and refines them iteratively.

---

## 🔄 Framework Pipeline

<p align="center">
  <img src="assets/framework.png" width="80%" />
</p>

The system operates in **three stages**:

### 1️⃣ Demonstration Mining (D<sub>GT</sub>)
- Generate candidate programs  
- Execute and verify outputs  
- Keep only **correct demonstrations**

### 2️⃣ Graph Memory + RL Training
- Build graph from demonstrations  
- Train RL policy for retrieval  

### 3️⃣ Inference
- Retrieve demonstrations  
- Generate Python code  
- Execute + debug  
- Produce final answer  

---

## ⚙️ Setup

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt

```

## 📍 How to Find Project

This project is located on the server at `~/code/ehr-rlgraph`. All datasets used in this work are available in the `~/code/ehr-rlgraph/data` directory. Before running any part of the pipeline, please activate the conda environment using `conda activate ehragent`. Once the environment is set up, you can proceed with executing the commands listed below.


## 🔑 Set API Keys

```bash
export OPENAI_API_KEY=your_api_key
export WOLFRAM_ALPHA_APPID=your_app_id  
```

---

## 📂 Supported Datasets

- `mimic_iii`
- `eicu`
- `treqs`

---

## 🚀 Running the Pipeline

### ▶️ Step 1: Build Demonstrations

```bash
python -m ehragent.build_dgt --dataset mimic_iii --llm gpt-4.1 --data_path <DATA_PATH> --out_path <DGT_JSONL> --graph_out <GRAPH_JSON> --logs_dir <LOG_DIR> --num_examples -1 --seed 42
```

### ▶️ Step 2: Train RL Retriever

```bash
python -m ehragent.train_rl --dgt_jsonl <DGT_JSONL> --graph_path <GRAPH_JSON> --k 4 --num_train <NUM_TRAIN> --seed 42 --save_policy <POLICY_JSON>
```

### ▶️ Step 3: Run Inference

```bash
python ehragent/main.py --llm gpt-4.1 --dataset mimic_iii --data_path <DATA_PATH> --logs_path <LOG_DIR> --dgt_jsonl <DGT_JSONL> --graph_path <GRAPH_JSON> --policy_path <POLICY_JSON> --num_questions <N> --num_shots 4 --seed 42
```

---

## 📊 Evaluation

```bash
python -m ehragent.evaluate --logs_path <LOG_DIR> --data_path <DATA_PATH> --id_to_level <LEVEL_JSON>
```

---

## 📈 Metrics

- **Success Rate (SR):** Correct answers / total queries
- **Completion Rate (CR):** Executable programs / total queries
- **Complexity-level performance:** SR and CR across difficulty levels

---

## 🧪 Ablation Settings

| Setting | Description |
|---|---|
| **Full Model** | Uses graph memory, RL policy, and demonstrations |
| **w/o RL** | Remove `--policy_path` |
| **w/o Graph** | Remove both `--graph_path` and `--policy_path` |
| **w/o Knowledge** | Add `--disable_knowledge` |

---

## 📁 Outputs

### 🧩 Demonstration Mining

- `dgt_*.jsonl` → execution-verified demonstrations
- `graph_*.json` → graph memory

### 🎯 RL Training

- `policy_*.json` → trained RL retriever

### 🧠 Inference

- `.txt` logs → per-query execution traces

---

## 🔍 How RL Helps

The RL stage **does not create new demonstrations**.

Instead, it learns:

- which demonstrations are useful
- how to balance relevance, diversity, and structure
- how to retrieve examples that better support code generation

At inference time, the trained policy helps select better demonstrations from the graph memory, which leads to stronger code generation and higher success on complex EHR queries.

---

## 📝 Notes

- All experiments use **GPT-4.1**
- Random seed is fixed for reproducibility
- Logs contain full execution traces for each query
- The graph and RL policy must be generated before running the full model

---

## 📌 Summary

EHR-RLGraph improves EHR reasoning by:

- replacing static retrieval with learned retrieval
- leveraging structured graph memory
- combining RL retrieval with execution-feedback-based code generation
