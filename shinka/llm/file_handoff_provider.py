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

The agent can write a pending/<id>.inprogress marker to signal it has
picked up the request. This resets the timeout deadline, giving the agent
the full timeout window from the acknowledgment point.
"""

import json
import os
import time
import uuid
from pathlib import Path


HANDOFF_DIR: Path | None = None

_DEFAULT_TIMEOUT = 600  # 10 minutes; override with SHINKA_HANDOFF_TIMEOUT env var
_HEARTBEAT_INTERVAL = 5  # seconds between heartbeat writes


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
    timeout_seconds: int | None = None,
    **kwargs,
) -> dict:
    """Write prompt to file, wait for orchestrator to dispatch and respond.

    Args:
        timeout_seconds: Max seconds to wait. If None, reads from
            SHINKA_HANDOFF_TIMEOUT env var, then falls back to _DEFAULT_TIMEOUT.
            When the agent writes a <id>.inprogress marker, the deadline
            resets from that point.

    Returns a dict compatible with ShinkaEvolve's QueryResult:
    {"content": str, "cost": float, "model_name": str, ...}
    """
    global HANDOFF_DIR
    if HANDOFF_DIR is None:
        env_dir = os.environ.get("SHINKA_HANDOFF_DIR")
        if env_dir:
            HANDOFF_DIR = set_handoff_dir(env_dir)
        else:
            raise RuntimeError(
                "Handoff directory not configured. Set SHINKA_HANDOFF_DIR env var "
                "or call set_handoff_dir() first."
            )

    # Resolve timeout: explicit param > env var > default
    if timeout_seconds is None:
        env_timeout = os.environ.get("SHINKA_HANDOFF_TIMEOUT")
        if env_timeout is not None:
            try:
                timeout_seconds = int(env_timeout)
            except ValueError:
                timeout_seconds = _DEFAULT_TIMEOUT
        else:
            timeout_seconds = _DEFAULT_TIMEOUT

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

    # Poll for response with deadline-based loop
    completed_path = HANDOFF_DIR / "completed" / f"{request_id}.json"
    heartbeat_path = HANDOFF_DIR / "pending" / f"{request_id}.heartbeat"
    inprogress_path = HANDOFF_DIR / "pending" / f"{request_id}.inprogress"

    start_time = time.monotonic()
    deadline = start_time + timeout_seconds
    last_heartbeat = 0.0
    deadline_extended = False

    while True:
        now = time.monotonic()

        # Check if agent acknowledged — extend deadline once
        if not deadline_extended and inprogress_path.exists():
            deadline = now + timeout_seconds
            deadline_extended = True

        if now > deadline:
            break

        if completed_path.exists():
            try:
                response = json.loads(completed_path.read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            # Cleanup all files
            pending_path.unlink(missing_ok=True)
            completed_path.unlink(missing_ok=True)
            heartbeat_path.unlink(missing_ok=True)
            inprogress_path.unlink(missing_ok=True)
            return {
                "content": response.get("content", ""),
                "msg": msg,
                "system_msg": system_msg,
                "new_msg_history": [],
                "kwargs": kwargs,
                "cost": 0.0,
                "model_name": model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "model_posteriors": {},
                "num_tool_calls": 0,
            }

        # Write heartbeat so agent knows we're still waiting
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
            try:
                heartbeat_path.write_text(json.dumps({
                    "elapsed": round(now - start_time, 1),
                    "remaining": round(deadline - now, 1),
                    "status": "waiting",
                }))
            except OSError:
                pass
            last_heartbeat = now

        time.sleep(1)

    # Timeout — cleanup
    pending_path.unlink(missing_ok=True)
    heartbeat_path.unlink(missing_ok=True)
    inprogress_path.unlink(missing_ok=True)
    raise TimeoutError(
        f"No response for request {request_id} after {timeout_seconds}s"
    )
