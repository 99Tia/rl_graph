from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Any, Tuple
import json
import os
import numpy as np
import re


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = logits.astype(np.float64)
    t = float(temperature)
    if t <= 0:
        t = 1e-6
    x = x / t
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x, dtype=np.float64) / max(1, len(x))
    return ex / s

# some helper functions for extra rl features, added for treqs
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0

def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _tokenize(text: str) -> Set[str]:
    text = _normalize_text(text)
    if not text:
        return set()
    return set(text.split())

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return _safe_div(len(a & b), len(a | b))

def _extract_code_like_tokens(text: str) -> Set[str]:
    text = (text or "").lower()
    out = set()

    for tok in re.findall(r"\b\d+\b", text):
        out.add(tok)

    for tok in re.findall(r"\b[a-z]*\d+[a-z0-9]*\b", text):
        out.add(tok)

    special_keywords = [
        "icd9", "itemid", "drug", "route", "diagnosis", "procedure",
        "lab", "insurance", "ethnicity", "language", "gender",
        "subject_id", "hadm_id", "dob_year", "dod_year", "admityear"
    ]
    for kw in special_keywords:
        if kw in text:
            out.add(kw)

    return out

def _infer_question_type(text: str) -> str:
    t = _normalize_text(text)

    if any(x in t for x in ["how many", "number of", "count ", "total number", "total patients"]):
        return "count"
    if any(x in t for x in ["average", "mean"]):
        return "mean"
    if "maximum" in t or "max " in t or "highest" in t:
        return "max"
    if "minimum" in t or "min " in t or "lowest" in t:
        return "min"
    if any(x in t for x in ["list", "which patients", "what are", "show me all"]):
        return "list"
    return "lookup"

def _infer_answer_style_from_question(text: str) -> str:
    t = _normalize_text(text)

    if any(x in t for x in ["how many", "number of", "count ", "average", "mean", "minimum", "maximum", "min ", "max ", "total number"]):
        return "scalar"

    if any(x in t for x in ["which patients", "list", "what are", "show me all"]):
        return "list"

    # heuristic: many "what is ..." questions are lookups, not lists
    if "what is" in t or "tell me" in t or "provide" in t or "specify" in t:
        return "lookup"

    return "lookup"

def _tool_flags_from_code(code: str) -> Dict[str, float]:
    c = code or ""
    return {
        "uses_sql": 1.0 if "SQLInterpreter(" in c else 0.0,
        "uses_loaddb": 1.0 if "LoadDB(" in c else 0.0,
        "uses_filterdb": 1.0 if "FilterDB(" in c else 0.0,
        "uses_getvalue": 1.0 if "GetValue(" in c else 0.0,
        "uses_calendar": 1.0 if "Calendar(" in c else 0.0,
        "uses_calculate": 1.0 if "Calculate(" in c else 0.0,
    }

def _overlap_ratio(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return _safe_div(len(a & b), min(len(a), len(b)))


@dataclass
class RLRetrieverConfig:
    k_demos: int = 4
    seed_top_m: int = 20
    expand_hops: int = 1
    lr: float = 0.05
    entropy_bonus: float = 0.01
    baseline_momentum: float = 0.9
    max_candidates: int = 200
    temperature: float = 1.0
    greedy_inference: bool = False

class RLRetriever:
    def __init__(
        self,
        graph,
        config: Optional[RLRetrieverConfig] = None,
        rng_seed: int = 0,
    ):
        self.graph = graph
        self.cfg = config or RLRetrieverConfig()
        self.rng = np.random.default_rng(int(rng_seed))
   
        self._feature_dim = 10
        self.w = np.zeros((self._feature_dim,), dtype=np.float64)
        self.baseline = 0.0

        self._last_steps: Optional[List[Tuple[np.ndarray, np.ndarray, int]]] = None
        self._last_entropy: float = 0.0


    def set_eval_mode(self) -> None:
        self.cfg.greedy_inference = True
        if self.cfg.temperature > 1e-3:
            self.cfg.temperature = 1e-3

    def set_train_mode(self, temperature: Optional[float] = None, greedy_inference: bool = False) -> None:
        self.cfg.greedy_inference = bool(greedy_inference)
        if temperature is not None:
            self.cfg.temperature = float(temperature)
    
    def _query_profile(self, query: str) -> Dict[str, Any]:
        tokens = _tokenize(query)
        code_tokens = _extract_code_like_tokens(query)
        qtype = _infer_question_type(query)
        ans_style = _infer_answer_style_from_question(query)
        return {
            "tokens": tokens,
            "code_tokens": code_tokens,
            "qtype": qtype,
            "ans_style": ans_style,
        }
    
    def _node_profile(self, idx: int) -> Dict[str, Any]:
        n = self.graph.get_node(idx)

        q = getattr(n, "q", "") or ""
        p = getattr(n, "p", "") or ""
        s = getattr(n, "s", {}) or {}
        c = getattr(n, "c", "") or ""

        q_tokens = _tokenize(q)
        q_code_tokens = _extract_code_like_tokens(q)

        plan_tokens = _tokenize(p)
        table_set = set((s.get("tables") or [])) if isinstance(s.get("tables"), list) else set()
        col_set = set((s.get("columns") or [])) if isinstance(s.get("columns"), list) else set()

        qtype = _infer_question_type(q)
        ans_style = _infer_answer_style_from_question(q)
        tool_flags = _tool_flags_from_code(c)

        return {
            "q_tokens": q_tokens,
            "q_code_tokens": q_code_tokens,
            "plan_tokens": plan_tokens,
            "table_set": table_set,
            "col_set": col_set,
            "qtype": qtype,
            "ans_style": ans_style,
            "tool_flags": tool_flags,
        }

    def _query_table_hints(self, query: str) -> Set[str]:
        q = (query or "").lower()
        hints: Set[str] = set()

        if any(x in q for x in ["diagnosis", "icd9", "short title", "long title"]):
            hints.add("DIAGNOSES")
        if any(x in q for x in ["procedure", "procedures"]):
            hints.add("PROCEDURES")
        if any(x in q for x in ["drug", "route", "prescription", "formulary"]):
            hints.add("PRESCRIPTIONS")
        if any(x in q for x in ["lab", "itemid", "label", "fluid", "category"]):
            hints.add("LAB")
        if any(x in q for x in [
            "age", "gender", "language", "religion", "insurance", "ethnicity",
            "admission", "discharge", "marital", "born", "dob", "died", "death",
            "primary disease", "diagnosis"
        ]):
            hints.add("DEMOGRAPHIC")

        return hints

    def _query_column_hints(self, query: str) -> Set[str]:
        q = (query or "").lower()
        cols: Set[str] = set()

        mapping = {
            "age": "AGE",
            "gender": "GENDER",
            "language": "LANGUAGE",
            "religion": "RELIGION",
            "insurance": "INSURANCE",
            "ethnicity": "ETHNICITY",
            "born": "DOB_YEAR",
            "birth": "DOB_YEAR",
            "died": "DOD_YEAR",
            "death": "DOD_YEAR",
            "admission type": "ADMISSION_TYPE",
            "admission": "ADMITTIME",
            "discharge": "DISCHTIME",
            "days": "DAYS_STAY",
            "stay": "DAYS_STAY",
            "primary disease": "DIAGNOSIS",
            "drug code": "FORMULARY_DRUG_CD",
            "drug dose": "DRUG_DOSE",
            "drug route": "ROUTE",
            "drug name": "DRUG",
            "route": "ROUTE",
            "itemid": "ITEMID",
            "label": "LABEL",
            "fluid": "FLUID",
            "category": "CATEGORY",
            "flag": "FLAG",
            "icd9": "ICD9_CODE",
            "short title": "SHORT_TITLE",
            "long title": "LONG_TITLE",
        }

        for key, val in mapping.items():
            if key in q:
                cols.add(val)

        return cols

    def _features(self, q_emb: np.ndarray, query: str, idx: int, chosen: Set[int]) -> np.ndarray:
        e_i = self.graph.node_embedding(idx)
        cos_qi = float(np.dot(q_emb, e_i))

        if not chosen:
            max_sim_to_chosen = 0.0
        else:
            max_sim_to_chosen = max(float(np.dot(e_i, self.graph.node_embedding(j))) for j in chosen)

        deg = float(len(self.graph.get_neighbors(idx)))
        deg = float(np.tanh(deg / 10.0))

        qp = self._query_profile(query)
        npf = self._node_profile(idx)

        lexical_overlap = _jaccard(qp["tokens"], npf["q_tokens"])
        code_token_overlap = _jaccard(qp["code_tokens"], npf["q_code_tokens"])

        qtype_match = 1.0 if qp["qtype"] == npf["qtype"] else 0.0
        ans_style_match = 1.0 if qp["ans_style"] == npf["ans_style"] else 0.0

        q_table_hints = self._query_table_hints(query)
        q_col_hints = self._query_column_hints(query)

        table_overlap = _overlap_ratio(q_table_hints, set(str(x) for x in npf["table_set"]))
        col_overlap = _overlap_ratio(q_col_hints, set(str(x) for x in npf["col_set"]))
        
        wants_sql = 1.0 if any(x in _normalize_text(query) for x in [
            "how many", "average", "minimum", "maximum", "join", "diagnosis", "procedure", "itemid", "drug code"
        ]) else 0.0
        tool_sql_match = 1.0 if (wants_sql > 0.0 and npf["tool_flags"]["uses_sql"] > 0.0) else 0.0

        plan_overlap = _jaccard(_tokenize(query), npf["plan_tokens"])

        return np.array([
            cos_qi,                      # 0 relevance
            lexical_overlap,             # 1 lexical overlap
            code_token_overlap,          # 2 exact/code-like overlap
            qtype_match,                 # 3 question family match
            ans_style_match,             # 4 scalar/list style match
            table_overlap,               # 5 table hint overlap
            col_overlap,                 # 6 column hint overlap
            plan_overlap,                # 7 plan overlap
            tool_sql_match,              # 8 rough tool compatibility
            -max_sim_to_chosen + deg,    # 9 diversity + generality prior
        ], dtype=np.float64)

    def _build_candidate_pool(self, query: str) -> List[int]:
        seed = self.graph.topM_seed(query, M=self.cfg.seed_top_m)
        pool: Set[int] = set(seed)
        frontier = set(seed)

        for _ in range(int(self.cfg.expand_hops)):
            nxt = set()
            for idx in frontier:
                for nb in self.graph.get_neighbors(idx):
                    nxt.add(nb)
            pool |= nxt
            frontier = nxt

        pool_list = list(pool)

        if len(pool_list) > int(self.cfg.max_candidates):
            q_emb = self.graph.query_embedding(query)
            scored = [(float(np.dot(q_emb, self.graph.node_embedding(i))), i) for i in pool_list]
            scored.sort(reverse=True)
            pool_list = [i for _, i in scored[: int(self.cfg.max_candidates)]]

        return pool_list

    def select(self, query: str, k: Optional[int] = None) -> List[int]:
        k = int(k or self.cfg.k_demos)
        candidates = self._build_candidate_pool(query)

        if not candidates or k <= 0:
            self._last_steps = None
            self._last_entropy = 0.0
            return []

        q_emb = self.graph.query_embedding(query)
        chosen: Set[int] = set()
        chosen_list: List[int] = []
        steps: List[Tuple[np.ndarray, np.ndarray, int]] = []
        entropies: List[float] = []

        for _step in range(k):
            avail = [idx for idx in candidates if idx not in chosen]
            if not avail:
                break

            feat_mat = np.stack([self._features(q_emb, query, idx, chosen) for idx in avail], axis=0)  # (n,3)

             # expand weight if old checkpoint had smaller size
            if self.w.shape[0] != feat_mat.shape[1]:
                new_w = np.zeros((feat_mat.shape[1],), dtype=np.float64)
                copy_dim = min(self.w.shape[0], new_w.shape[0])
                new_w[:copy_dim] = self.w[:copy_dim]
                self.w = new_w

            logits = feat_mat @ self.w
            probs = _softmax(logits, temperature=self.cfg.temperature)

            if self.cfg.greedy_inference:
                a_pos = int(np.argmax(probs))
            else:
                a_pos = int(self.rng.choice(len(avail), p=probs))

            a_idx = avail[a_pos]
            chosen.add(a_idx)
            chosen_list.append(a_idx)

            steps.append((feat_mat, probs, a_pos))
            ent = -float(np.sum(probs * np.log(probs + 1e-12)))
            entropies.append(ent)

        self._last_steps = steps
        self._last_entropy = float(np.mean(entropies)) if entropies else 0.0
        return chosen_list

    def update(self, reward: float) -> None:
        if not self._last_steps:
            return

        r = float(reward)
        self.baseline = (
            float(self.cfg.baseline_momentum) * float(self.baseline)
            + (1.0 - float(self.cfg.baseline_momentum)) * r
        )
        adv = float(r - self.baseline)

        grad = np.zeros_like(self.w, dtype=np.float64)
        for feat_mat, probs, a_pos in self._last_steps:
            exp_feat = probs @ feat_mat
            act_feat = feat_mat[a_pos]
            grad += (act_feat - exp_feat)

        grad *= adv

        if float(self.cfg.entropy_bonus) > 0:
            ent = float(self._last_entropy)
            ent_grad = -self.w * max(0.0, (1.0 - ent))
            grad += float(self.cfg.entropy_bonus) * ent_grad

        self.w += float(self.cfg.lr) * grad

        self._last_steps = None
        self._last_entropy = 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            "w": self.w.tolist(),
            "baseline": float(self.baseline),
            "feature_dim": int(self.w.shape[0]),
            "cfg": {
                "k_demos": int(self.cfg.k_demos),
                "seed_top_m": int(self.cfg.seed_top_m),
                "expand_hops": int(self.cfg.expand_hops),
                "lr": float(self.cfg.lr),
                "entropy_bonus": float(self.cfg.entropy_bonus),
                "baseline_momentum": float(self.cfg.baseline_momentum),
                "max_candidates": int(self.cfg.max_candidates),
                "temperature": float(self.cfg.temperature),
                "greedy_inference": bool(self.cfg.greedy_inference),
            },
        }

    def load_state(self, state: Dict[str, Any], load_cfg: bool = True) -> None:
        if not isinstance(state, dict):
            return

        if "w" in state:
            self.w = np.array(state["w"], dtype=np.float64)

        if "baseline" in state:
            self.baseline = float(state["baseline"])

        if load_cfg and isinstance(state.get("cfg"), dict):
            cfgd = state["cfg"]

            if "k_demos" in cfgd:
                self.cfg.k_demos = int(cfgd["k_demos"])
            if "seed_top_m" in cfgd:
                self.cfg.seed_top_m = int(cfgd["seed_top_m"])
            if "expand_hops" in cfgd:
                self.cfg.expand_hops = int(cfgd["expand_hops"])
            if "lr" in cfgd:
                self.cfg.lr = float(cfgd["lr"])
            if "entropy_bonus" in cfgd:
                self.cfg.entropy_bonus = float(cfgd["entropy_bonus"])
            if "baseline_momentum" in cfgd:
                self.cfg.baseline_momentum = float(cfgd["baseline_momentum"])
            if "max_candidates" in cfgd:
                self.cfg.max_candidates = int(cfgd["max_candidates"])
            if "temperature" in cfgd:
                self.cfg.temperature = float(cfgd["temperature"])
            if "greedy_inference" in cfgd:
                self.cfg.greedy_inference = bool(cfgd["greedy_inference"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f, indent=2)
        print(f"[INFO] Saved RLRetriever state -> {path}")

    def load_from_file(self, path: str, load_cfg: bool = True) -> None:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state(state, load_cfg=load_cfg)

    @classmethod
    def from_file(
        cls,
        graph,
        path: str,
        config: Optional[RLRetrieverConfig] = None,
        rng_seed: int = 0,
        load_cfg: bool = True,
    ) -> "RLRetriever":
        rr = cls(graph=graph, config=config, rng_seed=rng_seed)
        rr.load_from_file(path, load_cfg=load_cfg)
        return rr