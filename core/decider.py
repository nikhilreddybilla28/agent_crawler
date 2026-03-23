"""LLM integration for action selection with telemetry tracking."""

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field

from .observer import InteractiveElement


# ---------------------------------------------------------------------------
# Telemetry — track LLM call metrics for observability
# ---------------------------------------------------------------------------

@dataclass
class LLMCallRecord:
    timestamp: float
    provider: str
    model: str
    prompt_purpose: str         # "action_selection" | "login"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: str = ""


_telemetry: list[LLMCallRecord] = []


def get_telemetry() -> dict:
    """Return telemetry summary for all LLM calls in this session."""
    total = len(_telemetry)
    successes = sum(1 for r in _telemetry if r.success)
    total_input = sum(r.input_tokens for r in _telemetry)
    total_output = sum(r.output_tokens for r in _telemetry)
    total_latency = sum(r.latency_ms for r in _telemetry)
    return {
        "total_calls": total,
        "successful_calls": successes,
        "failed_calls": total - successes,
        "success_rate": successes / total if total else 0,
        "total_tokens": total_input + total_output,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_latency_ms": round(total_latency, 1),
        "avg_latency_ms": round(total_latency / total, 1) if total else 0,
        "calls": [
            {
                "provider": r.provider,
                "model": r.model,
                "purpose": r.prompt_purpose,
                "tokens": r.input_tokens + r.output_tokens,
                "latency_ms": round(r.latency_ms, 1),
                "success": r.success,
                "error": r.error,
            }
            for r in _telemetry
        ],
    }


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    element_index: int
    action_type: str       # "click", "fill", "select", etc.
    value: str | None      # for fill/select actions
    reasoning: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a web exploration agent. Your job is to explore a web application by interacting with UI elements to discover all distinct states/pages.

You will be given:
1. The current page URL and title
2. A numbered list of interactive elements on the page
3. A list of elements you have already interacted with on this page
4. (Optional) Information-gain hints showing which element types have been most productive

Your task: Choose ONE element to interact with that is most likely to lead to a new, unexplored state. Prefer navigation links, tabs, and buttons that suggest new pages/views over elements that would repeat or stay on the same page.

Rules:
- Do NOT choose elements you have already explored (listed in "Already Explored")
- Prefer links and navigation buttons over form inputs (unless filling a form is needed to proceed)
- Avoid destructive actions: do not click "delete", "remove", "logout", "sign out", "reset" buttons
- If you see a login form and credentials are provided, fill in the form fields first
- Consider the information-gain hints: higher scores mean that element type has historically led to new states

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "element_index": <int>,
  "action_type": "click" | "fill" | "select",
  "value": "<string or null>",
  "reasoning": "<brief explanation>"
}\
"""

LOGIN_SYSTEM_PROMPT = """\
You are a web agent that needs to log in to a web application. You will be given the page context and credentials.

Look at the interactive elements and determine which ones are the username/email field, password field, and submit button. Then provide a sequence of actions to fill in credentials and submit.

Respond with ONLY a JSON array of actions (no markdown):
[
  {"element_index": <int>, "action_type": "fill", "value": "<username>"},
  {"element_index": <int>, "action_type": "fill", "value": "<password>"},
  {"element_index": <int>, "action_type": "click", "value": null}
]\
"""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _get_llm_client():
    """Get LLM client — supports VT ARC LLM API, Anthropic, and OpenAI."""
    arc_key = os.environ.get("ARC_API_KEY")
    if arc_key:
        import openai
        return "arc", openai.OpenAI(
            api_key=arc_key,
            base_url="https://llm-api.arc.vt.edu/api/v1",
        )

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=anthropic_key)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        import openai
        return "openai", openai.OpenAI(api_key=openai_key)

    raise RuntimeError(
        "No LLM API key found. Set ARC_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


def _call_llm(system: str, user: str, purpose: str = "action_selection") -> str:
    """Call the LLM and return the response text. Records telemetry."""
    provider, client = _get_llm_client()
    start = time.time()
    input_tokens = 0
    output_tokens = 0
    model_used = ""
    error_msg = ""
    success = False
    result = ""

    try:
        if provider == "anthropic":
            model_used = "claude-sonnet-4-20250514"
            response = client.messages.create(
                model=model_used,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            result = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        else:
            model_used = os.environ.get("LLM_MODEL", "gpt-oss-120b" if provider == "arc" else "gpt-4o-mini")
            print(f"[INFO] : calling {model_used} from {provider}")
            response = client.chat.completions.create(
                model=model_used,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            result = response.choices[0].message.content
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
            print(f"[INFO] : model responded :{result[:10]}")

        success = True

    except Exception as e:
        error_msg = str(e)
        raise

    finally:
        elapsed = (time.time() - start) * 1000
        _telemetry.append(LLMCallRecord(
            timestamp=start,
            provider=provider,
            model=model_used,
            prompt_purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            success=success,
            error=error_msg,
        ))

    return result


def _parse_json(text: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Action decision
# ---------------------------------------------------------------------------

def decide_action(
    page_context: str,
    explored_indices: set[int],
    elements: list[InteractiveElement],
    role_rewards: dict[str, float] | None = None,
) -> AgentAction | None:
    """Ask the LLM which element to interact with next.

    Optionally accepts role_rewards — a dict mapping element roles to
    information-gain scores (higher = more likely to discover new states).
    These are appended to the prompt as exploration hints.
    """
    if not elements:
        return None

    explored_desc = []
    for idx in explored_indices:
        if idx < len(elements):
            explored_desc.append(f"  [{idx}] {elements[idx].description}")

    user_msg = page_context
    if explored_desc:
        user_msg += "\n\nAlready Explored:\n" + "\n".join(explored_desc)
    else:
        user_msg += "\n\nAlready Explored: (none)"

    # Add information-gain hints
    if role_rewards:
        hints = sorted(role_rewards.items(), key=lambda x: x[1], reverse=True)
        hint_lines = [f"  {role}: {score:.2f}" for role, score in hints[:6]]
        user_msg += "\n\nInformation-Gain Scores (higher = more likely to find new states):\n" + "\n".join(hint_lines)

    available = [e for e in elements if e.index not in explored_indices]
    if not available:
        return None

    try:
        response = _call_llm(SYSTEM_PROMPT, user_msg, purpose="action_selection")
        data = _parse_json(response)
        return AgentAction(
            element_index=int(data["element_index"]),
            action_type=data.get("action_type", "click"),
            value=data.get("value"),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  [warn] LLM parse failed ({e}), using fallback selection")
        # Fallback: pick unexplored element with highest information-gain score
        if role_rewards:
            available.sort(key=lambda el: role_rewards.get(el.role, 0.5), reverse=True)
        el = available[0]
        return AgentAction(
            element_index=el.index,
            action_type="click",
            value=None,
            reasoning="fallback — LLM response parsing failed, picked by info-gain score",
        )


def decide_login_actions(
    page_context: str,
    username: str,
    password: str,
) -> list[AgentAction]:
    """Ask the LLM to figure out how to fill in login credentials."""
    user_msg = (
        f"{page_context}\n\n"
        f"Credentials:\n  Username: {username}\n  Password: {password}"
    )

    try:
        response = _call_llm(LOGIN_SYSTEM_PROMPT, user_msg, purpose="login")
        data = _parse_json(response)
        actions = []
        for item in data:
            actions.append(AgentAction(
                element_index=int(item["element_index"]),
                action_type=item.get("action_type", "click"),
                value=item.get("value"),
                reasoning="login sequence",
            ))
        return actions
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  [warn] Login LLM parse failed: {e}")
        return []
