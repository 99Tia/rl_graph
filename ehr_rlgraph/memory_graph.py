from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from ehr_rlgraph.code_parser import parse_code_cell


@dataclass
class MemoryNode:
    q: str                          # question text
    c: str                          # executable code
    p: str                          # plan skeleton 
    s: Dict[str, Any]               # schema footprint
    e: List[float]                  # embedding vector
    t: float                        # timestamp
    meta: Dict[str, Any] = field(default_factory=dict)


class MiniLMEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        emb = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


class GraphMemory:
    def __init__(self, k_neighbors: int = 5, encoder_name: str = "all-MiniLM-L6-v2"):
        self.k = int(k_neighbors)
        self.encoder_name = encoder_name
        self.encoder = MiniLMEncoder(encoder_name)

        self.nodes: List[MemoryNode] = []
        self.neighbors: List[List[Tuple[int, float]]] = []  
        self._E: Optional[np.ndarray] = None  
    #  helpers (added later)
    def num_nodes(self) -> int:
        return len(self.nodes)

    def degree(self, idx: int) -> int:
        if idx < 0 or idx >= len(self.neighbors):
            return 0
        return len(self.neighbors[idx])

    def get_node(self, idx: int) -> MemoryNode:
        return self.nodes[idx]

    def _node_text(self, q: str, p: str, s: Dict[str, Any]) -> str:
        tables = ", ".join((s.get("tables") or []) if isinstance(s.get("tables"), list) else [])
        cols = ", ".join((s.get("columns") or []) if isinstance(s.get("columns"), list) else [])
        return f"Q: {q}\nP: {p}\nTABLES: {tables}\nCOLS: {cols}"

    def rebuild(self) -> None:
        n = len(self.nodes)
        if n == 0:
            self._E = None
            self.neighbors = []
            return

        first_text = self._node_text(self.nodes[0].q, self.nodes[0].p, self.nodes[0].s)
        first_e = self.encoder.encode(first_text)
        dim = int(first_e.shape[0])

        E = np.zeros((n, dim), dtype=np.float32)
        E[0] = first_e
        self.nodes[0].e = first_e.tolist()

        for i in range(1, n):
            node = self.nodes[i]
            text = self._node_text(node.q, node.p, node.s)
            e = self.encoder.encode(text)
            if e.shape[0] != dim:
                raise RuntimeError(f"Embedding dim mismatch: got {e.shape[0]}, expected {dim}")
            node.e = e.tolist()
            E[i] = e

        self._E = E
        self.neighbors = [[] for _ in range(n)]
        sims_full = E @ E.T 

        for i in range(n):
            sims_full[i, i] = -1.0  
            topk = min(self.k, n - 1)
            if topk <= 0:
                continue
            idxs = np.argsort(-sims_full[i])[:topk]
            self.neighbors[i] = [(int(j), float(sims_full[i, j])) for j in idxs]

    def add_success_case(self, question: str, code: str, *, meta: Optional[Dict[str, Any]] = None, rebuild: bool = True,) -> int:
        parsed = parse_code_cell(code)
        node = MemoryNode(
            q=question,
            c=code,
            p=parsed.get("plan_skeleton", "None") or "None",
            s=parsed.get("schema_footprint", {"tables": [], "columns": []}) or {"tables": [], "columns": []},
            e=[],
            t=time.time(),
            meta=meta or {},
        )
        self.nodes.append(node)
        if rebuild:
            self.rebuild()
        return len(self.nodes) - 1

    def build_from_memory_list(self, memory_list: List[Dict[str, Any]], *, rebuild: bool = True) -> None:
        self.nodes = []
        for item in memory_list:
            q = item.get("question", "") or ""
            c = item.get("code", "") or ""

            meta = dict(item)
            meta.pop("question", None)
            meta.pop("code", None)

            self.add_success_case(q, c, meta=meta, rebuild=False)

        if rebuild:
            self.rebuild()

    def query_embedding(self, query: str) -> np.ndarray:
        return self.encoder.encode(f"Q: {query}")

    def node_embedding(self, idx: int) -> np.ndarray:
        if self._E is None:
            raise RuntimeError("GraphMemory is empty. Add nodes first and rebuild().")
        return self._E[idx]

    def get_neighbors(self, idx: int) -> List[int]:
        if idx < 0 or idx >= len(self.neighbors):
            return []
        return [j for j, _ in self.neighbors[idx]]

    def topM_seed(self, query: str, M: int = 10) -> List[int]:
        if not self.nodes or self._E is None:
            return []
        eq = self.query_embedding(query)     
        sims = self._E @ eq                 
        idxs = np.argsort(-sims)[: min(int(M), len(self.nodes))]
        return [int(i) for i in idxs]

    def save(self, path: str) -> None:
        payload = {
            "k_neighbors": self.k,
            "encoder_name": self.encoder_name,
            "nodes": [asdict(n) for n in self.nodes],
            "neighbors": [
                [{"to": j, "sim": float(sim)} for j, sim in lst]
                for lst in self.neighbors
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "GraphMemory":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        gm = GraphMemory(
            k_neighbors=int(payload.get("k_neighbors", 5)),
            encoder_name=str(payload.get("encoder_name", "all-MiniLM-L6-v2")),
        )

        gm.nodes = [MemoryNode(**nd) for nd in payload.get("nodes", [])]
        gm.rebuild()
        return gm
