"""File-based handoff provider for Claude Code subagent dispatch.

Instead of calling LLM APIs directly, writes prompts to a pending
directory and polls for responses from the Claude Code orchestrator.

Flow:
  1. ShinkaEvolve calls query_file_handoff(model, msg, system_msg)
  2. This writes {id, system_msg, user_msg} to experiments/evolve/pending/<id>.json
  3. Blocks polling for experiments/evolve/completed/<id>.json
  4. The orchestrator (running concurrently) picks up pending requests,
     dispatches a Claude Code Agent() with the prompt, and writes the
     response to completed/<id>.json
  5. This function reads the response and returns it
"""

import json
import time
import uuid
from pathlib import Path


HANDOFF_DIR: Path | None = None


def set_handoff_dir(path: str) -> Path:
    """Configure the handoff directory. Call before starting evolution."""
    global HANDOFF_DIR
    HANDOFF_DIR = Path(path) / "evolve"
    (HANDOFF_DIR / "pending").mkdir(parents=True, exist_ok=True)
    (HANDOFF_DIR / "completed").mkdir(parents=True, exist_ok=True)
    return HANDOFF_DIR


def query_file_handoff(
    model_name: str,
    msg: str,
    system_msg: str,
    timeout_seconds: int = 300,
    **kwargs,
) -> dict:
    """Write prompt to file, wait for orchestrator to dispatch and respond.

    Returns a dict compatible with ShinkaEvolve's QueryResult:
    {"content": str, "cost": float, "model_name": str, ...}
    """
    if HANDOFF_DIR is None:
        raise RuntimeError(
            "Handoff directory not configured. Call set_handoff_dir() first."
        )

    request_id = uuid.uuid4().hex[:8]

    # Write request
    request = {
        "id": request_id,
        "system_msg": system_msg,
        "user_msg": msg,
        "model_name": model_name,
    }
    pending_path = HANDOFF_DIR / "pending" / f"{request_id}.json"
    pending_path.write_text(json.dumps(request, indent=2))

    # Poll for response
    completed_path = HANDOFF_DIR / "completed" / f"{request_id}.json"
    for _ in range(timeout_seconds):
        if completed_path.exists():
            try:
                response = json.loads(completed_path.read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            # Cleanup
            pending_path.unlink(missing_ok=True)
            completed_path.unlink(missing_ok=True)
            return {
                "content": response.get("content", ""),
                "cost": 0.0,
                "model_name": model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "model_posteriors": {},
                "num_tool_calls": 0,
            }
        time.sleep(1)

    # Timeout — cleanup pending request
    pending_path.unlink(missing_ok=True)
    raise TimeoutError(
        f"No response for request {request_id} after {timeout_seconds}s"
    )
