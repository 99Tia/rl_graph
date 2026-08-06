from __future__ import annotations
import os
from typing import Dict, Any

def get_backend(model_name: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please export it before running."
        )

    return {
        "api_type": "OPENAI",
        "model": model_name,   # e.g. gpt-4.1
        "api_key": api_key,
        "base_url": "https://api.openai.com/v1",
    }


def llm_config_list(seed: int, model_name: str) -> Dict[str, Any]:
    backend = get_backend(model_name)

    return {
        "functions": [
            {
                "name": "python",
                "description": "Run the entire code and return the execution result. Only generate code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {"type": "string", "description": "Valid Python code to execute."},
                        "code": {"type": "string", "description": "Alias of `cell` (valid Python code)."},
                    },
                    # not force "cell" required, otherwise AutoGen KeyError happens
                    "required": [],
                },
            }
        ],
        "config_list": [
            {
                "model": backend["model"],
                "api_key": backend["api_key"],
                "base_url": backend.get("base_url", "https://api.openai.com/v1"),
            }
        ],
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0.0,
    }
