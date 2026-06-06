from typing import List, Optional, Dict
from pydantic import BaseModel
from .client import get_client_llm, get_async_client_llm
from .providers import (
    query_anthropic,
    query_openai,
    query_deepseek,
    query_gemini,
    query_headless,
    query_local_openai,
    query_anthropic_async,
    query_openai_async,
    query_deepseek_async,
    query_gemini_async,
    query_headless_async,
    query_local_openai_async,
    QueryResult,
)
from .file_handoff_provider import query_file_handoff
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# Set SHINKA_PROVIDER=claude_code to route all LLM calls through
# Claude Code subagent file-based handoff instead of direct API calls.
_CLAUDE_CODE_MODE = os.environ.get("SHINKA_PROVIDER", "").lower() == "claude_code"


def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM."""
    # Claude Code subagent mode: bypass all API providers
    if _CLAUDE_CODE_MODE:
        result_dict = query_file_handoff(model_name, msg, system_msg, **kwargs)
        return QueryResult(**result_dict)

    client, model_name, provider = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    if provider in ("anthropic", "bedrock"):
        query_fn = query_anthropic
    elif provider in ("openai", "azure_openai", "openrouter"):
        query_fn = query_openai
    elif provider == "deepseek":
        query_fn = query_deepseek
    elif provider == "google":
        query_fn = query_gemini
    elif provider == "local_openai":
        query_fn = query_local_openai
    elif provider == "headless":
        query_fn = query_headless
    else:
        raise ValueError(f"Model {model_name} not supported.")
    result = query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )
    return result


async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM asynchronously."""
    # Claude Code subagent mode: run sync handoff in executor
    if _CLAUDE_CODE_MODE:
        loop = asyncio.get_event_loop()
        result_dict = await loop.run_in_executor(
            None, lambda: query_file_handoff(model_name, msg, system_msg, **kwargs)
        )
        return QueryResult(**result_dict)

    client, model_name, provider = get_async_client_llm(
        model_name, structured_output=output_model is not None
    )
    if provider in ("anthropic", "bedrock"):
        query_fn = query_anthropic_async
    elif provider in ("openai", "azure_openai", "openrouter"):
        query_fn = query_openai_async
    elif provider == "deepseek":
        query_fn = query_deepseek_async
    elif provider == "google":
        query_fn = query_gemini_async
    elif provider == "local_openai":
        query_fn = query_local_openai_async
    elif provider == "headless":
        query_fn = query_headless_async
    else:
        raise ValueError(f"Model {model_name} not supported.")
    result = await query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )
    return result
