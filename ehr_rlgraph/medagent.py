from __future__ import annotations
import time
import json
import logging
import inspect
from typing import Dict, List, Optional, Union, Callable, Literal, Any
from termcolor import colored
from autogen.agentchat import Agent, UserProxyAgent, ConversableAgent
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
import Levenshtein
from ehr_rlgraph.memory_graph import GraphMemory
from ehr_rlgraph.rl_retriever import RLRetriever, RLRetrieverConfig
logger = logging.getLogger(__name__)

RetrievalMode = Literal["levenshtein", "graph_rl"]


class MedAgent(UserProxyAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        config_list: Optional[List[Dict]] = None,
        debug: bool = False,
        use_llm_knowledge: bool = True,
        graph_path: str = "",
        k_neighbors: int = 5,
        retrieval_mode: RetrievalMode = "graph_rl",
        # for ablation study
        # disable_error_debugger: bool = False,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )

        self.config_list: List[Dict[str, Any]] = config_list or []
        self.dataset: str = "mimic_iii"
        self.question: str = ""
        self.code: str = ""
        self.knowledge: str = ""
        self.num_shots: int = 4
        self.memory: List[Dict[str, Any]] = []
        self.graph: Optional[GraphMemory] = None
        self.retriever: Optional[RLRetriever] = None
        self.graph_size: int = -1
        self._node2mem: List[int] = []
        self.debug = bool(debug)
        self.use_llm_knowledge = bool(use_llm_knowledge)
        self.graph_path = (graph_path or "").strip()
        self.k_neighbors = int(k_neighbors)
        self._using_prebuilt_graph: bool = False
        self.retrieval_mode: RetrievalMode = retrieval_mode
        # for ablation
        # self.disable_error_debugger = bool(disable_error_debugger)
        # For ablation study of without knowledge retrieval
        self.use_llm_knowledge = bool(use_llm_knowledge)

    def set_retrieval_mode(self, mode: RetrievalMode) -> None:
        self.retrieval_mode = mode
        if mode == "levenshtein":
            self.graph = None
            self.retriever = None
            self.graph_size = -1
            self._node2mem = []
            self._using_prebuilt_graph = False

    def _get_backend_config(self) -> Dict[str, Any]:
        if not self.config_list or not isinstance(self.config_list[0], dict):
            raise RuntimeError("MedAgent: config_list is empty or invalid")
        return self.config_list[0]

    def _chat_completion(
        self,
        cfg: Dict[str, Any],
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
    ) -> str:
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not available. Run: pip install openai")

        api_key = (cfg.get("api_key") or "").strip()
        if not api_key:
            raise RuntimeError("Missing OpenAI api_key in config_list")

        base_url = (cfg.get("base_url") or "").strip()
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

        model = cfg.get("model", "gpt-4.1")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=max_tokens,
            top_p=float(cfg.get("top_p", 0.95)),
        )
        return (resp.choices[0].message.content or "").strip()

    def retrieve_knowledge(self, config: Dict[str, Any], query: str) -> str:
        if not self.use_llm_knowledge:
            return ""

        if self.dataset == "mimic_iii":
            from ehr_rlgraph.prompts_mimic import RetrKnowledge
        elif self.dataset == "eicu":
            from ehr_rlgraph.prompts_eicu import RetrKnowledge
        else:
            from ehr_rlgraph.prompts_treqs import RetrKnowledge

        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": RetrKnowledge.format(question=query)},
        ]

        patience = 2
        last_err: Optional[Exception] = None

        while patience > 0:
            patience -= 1
            try:
                out = self._chat_completion(config, messages, max_tokens=800)
                if out:
                    return out
            except Exception as e:
                last_err = e
                logger.warning("retrieve_knowledge failed: %s", e)
                time.sleep(10)

        return f"Fail to retrieve related knowledge. Last error: {last_err}"

    def error_debugger(self, config: Dict[str, Any], code: str, error_info: str) -> str:
        if self.dataset == "mimic_iii":
            from ehr_rlgraph.prompts_mimic import CodeDebugger
        elif self.dataset == "eicu":
            from ehr_rlgraph.prompts_eicu import CodeDebugger
        else:
            from ehr_rlgraph.prompts_treqs import CodeDebugger

        max_len = 3000
        code = (code or "")[-max_len:]
        error_info = (error_info or "")[-max_len:]

        messages = [
            {"role": "system", "content": "You are an AI assistant that helps debug code. Give one likely reason."},
            {"role": "user", "content": CodeDebugger.format(question=self.question, code=code, error_info=error_info)},
        ]
        try:
            return self._chat_completion(config, messages, max_tokens=300)
        except Exception as e:
            return f"debugger failed: {e}"

    # will be added in build-dgt.py (stage 1)
    def _retrieve_examples_levenshtein(self, query: str) -> str:
        if not self.memory:
            return ""
        dists = []
        for i, m in enumerate(self.memory):
            q = (m.get("question") or "")
            dists.append((i, Levenshtein.distance(query, q)))
        dists.sort(key=lambda x: x[1])
        picked = [i for i, _ in dists[: min(self.num_shots, len(dists))]]

        examples = []
        for i in picked:
            mi = self.memory[i]
            template = (
                f"Question: {mi.get('question','')}\n"
                f"Knowledge:\n{mi.get('knowledge','')}\n"
                f"Solution:\n{mi.get('code','')}\n"
            )
            examples.append(template)
        return "\n".join(examples)

    def _load_graph_if_needed(self) -> bool:
        if not self.graph_path:
            return False
        if self.graph is not None and self._using_prebuilt_graph:
            return True

        self.graph = GraphMemory.load(self.graph_path)
        self._using_prebuilt_graph = True

        cfg = RLRetrieverConfig(
            k_demos=self.num_shots,
            seed_top_m=20,
            expand_hops=1,
            lr=0.05,
            entropy_bonus=0.01,
        )
        self.retriever = RLRetriever(self.graph, config=cfg, rng_seed=0)

        if self.debug:
            print(f"[DEBUG] Loaded prebuilt graph from {self.graph_path}")
        return True

    def _ensure_graph_and_retriever(self) -> None:
        if self._load_graph_if_needed():
            return

        if self.graph is not None and self.graph_size == len(self.memory) and self.retriever is not None:
            return

        self.graph = GraphMemory(k_neighbors=self.k_neighbors)
        self._node2mem = []

        can_batch = hasattr(self.graph, "rebuild")
        for mem_idx, item in enumerate(self.memory):
            q = item.get("question", "") or ""
            c = item.get("code", "") or ""
            meta = dict(item)
            meta.pop("question", None)
            meta.pop("code", None)

            try:
                self.graph.add_success_case(q, c, meta=meta, rebuild=False)
            except TypeError:
                self.graph.add_success_case(q, c)

            self._node2mem.append(mem_idx)

        if can_batch:
            self.graph.rebuild()

        self.graph_size = len(self.memory)

        cfg = RLRetrieverConfig(
            k_demos=self.num_shots,
            seed_top_m=20,
            expand_hops=1,
            lr=0.05,
            entropy_bonus=0.01,
        )
        self.retriever = RLRetriever(self.graph, config=cfg, rng_seed=0)

    def _format_example_from_node(self, node_idx: int) -> str:
        n = self.graph.nodes[node_idx]
        k = ""
        if isinstance(getattr(n, "meta", None), dict):
            k = str(n.meta.get("knowledge", "") or "")
        return f"Question: {n.q}\nKnowledge:\n{k}\nSolution:\n{n.c}\n"
    
    # will be added in other stage
    def _retrieve_examples_graph_rl(self, query: str) -> str:
        self._ensure_graph_and_retriever()

        k = min(self.num_shots, len(self.graph.nodes)) if self.graph and getattr(self.graph, "nodes", None) else self.num_shots
        try:
            node_ids = self.retriever.select(query, k=k)
        except Exception:
            node_ids = list(range(min(k, len(self.graph.nodes))))

        return "\n".join(self._format_example_from_node(i)[:1500] for i in node_ids)

    def retrieve_examples(self, query: str) -> str:
        if self.retrieval_mode == "levenshtein":
            return self._retrieve_examples_levenshtein(query)
        return self._retrieve_examples_graph_rl(query)

    # def generate_init_message(self, **context):
    #     if self.dataset == "mimic_iii":
    #         from ehr_rlgraph.prompts_mimic import EHRAgent_Message_Prompt
    #     elif self.dataset == "eicu":
    #         from ehr_rlgraph.prompts_eicu import EHRAgent_Message_Prompt
    #     else:
    #         from ehr_rlgraph.prompts_treqs import EHRAgent_Message_Prompt

    #     self.question = context["message"]
    #     self.knowledge = self.retrieve_knowledge(self._get_backend_config(), self.question)
    #     examples = self.retrieve_examples(self.question)

    #     return EHRAgent_Message_Prompt.format(
    #         examples=examples,
    #         knowledge=self.knowledge,
    #         question=self.question,
    #     )

    def generate_init_message(self, **context):
        if self.dataset == "mimic_iii":
            from ehr_rlgraph.prompts_mimic import EHRAgent_Message_Prompt
        elif self.dataset == "eicu":
            from ehr_rlgraph.prompts_eicu import EHRAgent_Message_Prompt
        else:
            from ehr_rlgraph.prompts_treqs import EHRAgent_Message_Prompt

        self.question = context["message"]

        if self.dataset == "treqs":
            self.knowledge = (
                "TREQS strict hints:\n"
                "- Prefer benchmark-style exact SQL.\n"
                "- If the question asks for one number, return one scalar only.\n"
                "- If the question says 'how many patients', usually use COUNT(DISTINCT SUBJECT_ID).\n"
                "- If the question says admissions / hospital admissions, think carefully whether HADM_ID is the correct unit.\n"
                "- If the question says 'primary disease', prefer DEMOGRAPHIC.DIAGNOSIS first.\n"
                "- Prefer exact equality before LIKE.\n"
                "- For codes, always use exact equality.\n"
                "- If a DB value is uncertain (for example inpatient, transfer wording, title wording, label wording), inspect exact DB values first with SQLInterpreter(select distinct ...).\n"
                "- Use SUBJECT_ID and HADM_ID to join tables when needed.\n"
                "- Do not invent hidden meanings for values.\n"
                "- When counting patients across multiple tables, use COUNT(DISTINCT SUBJECT_ID).\n"
                "- If joins are used, be careful about duplicate rows; DISTINCT may be needed.\n"
                "- If the question expects a single number, ensure the final result is a scalar number, not a list.\n"
                "- Prefer exact equality (=) for codes and categorical values instead of LIKE.\n"
                "- When counting admissions, use COUNT(DISTINCT HADM_ID).\n"
            )
        else:
            self.knowledge = self.retrieve_knowledge(self._get_backend_config(), self.question)

        examples = self.retrieve_examples(self.question)

        return EHRAgent_Message_Prompt.format(
            examples=examples,
            knowledge=self.knowledge,
            question=self.question,
        )

    def execute_function(self, func_call: Dict[str, Any]):
        func_name = func_call.get("name", "")
        raw_args = func_call.get("arguments", "{}")

        func = self._function_map.get(func_name)
        if func is None:
            content = f"Error: Function {func_name} not found."
            return False, {"name": func_name, "role": "function", "content": content}

        def _maybe_json_loads(x: Any) -> Any:
            if isinstance(x, dict):
                return x
            if not isinstance(x, str):
                return x
            s = x.strip()
            try:
                return json.loads(s)
            except Exception:
                return x

        args_obj = raw_args if isinstance(raw_args, dict) else _maybe_json_loads(raw_args)
        if not isinstance(args_obj, dict):
            args_obj = {"cell": str(raw_args)}

        arguments: Dict[str, Any] = {}
        for k, v in args_obj.items():
            kk = str(k).strip()
            if len(kk) >= 2 and kk[0] == kk[-1] and kk[0] in ("'", '"'):
                kk = kk[1:-1].strip()
            arguments[kk] = v

        if "cell" not in arguments:
            if "code" in arguments:
                arguments["cell"] = arguments.pop("code")
            else:
                arguments["cell"] = str(raw_args)

        cell = arguments.get("cell", "")
        if not isinstance(cell, str):
            cell = str(cell)
        arguments["cell"] = cell
        self.code = cell

        print(colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"), flush=True)

        try:
            sig = inspect.signature(func)
            if "dataset" in sig.parameters:
                arguments.setdefault("dataset", self.dataset)
            else:
                arguments.pop("dataset", None)
        except Exception:
            arguments.pop("dataset", None)

        ok = False
        try:
            out = func(**arguments)
            content = str(out)
            ok = True
        except Exception as e:
            content = f"Error: {type(e).__name__}: {e}"
            ok = False

        if ("error" in content.lower()) or ("traceback" in content.lower()):
            reasons = self.error_debugger(self._get_backend_config(), self.code, content)
            content = content + "\nPotential Reasons: " + reasons
        
        # For ablation study
        # if (("error" in content.lower()) or ("traceback" in content.lower())) and (not self.disable_error_debugger):
        #     reasons = self.error_debugger(self._get_backend_config(), self.code, content)
        #     content = content + "\nPotential Reasons: " + reasons

        return ok, {"name": func_name, "role": "function", "content": content}

    def update_memory(self, num_shots, memory):
        self.num_shots = int(num_shots)
        self.memory = memory or []
        self.graph_size = -1 

    def register_dataset(self, dataset):
        self.dataset = dataset