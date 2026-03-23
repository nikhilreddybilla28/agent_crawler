"""LLM integration for action selection with telemetry tracking."""

from __future__ import annotations
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .observer import InteractiveElement

# Module-level flag: set to True to send screenshots alongside text
VISION_ENABLED = False


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

SYSTEM_PROMPT_TEXT = """\
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

SYSTEM_PROMPT_VISION = """\
You are a web exploration agent. Your job is to explore a web application by interacting with UI elements to discover all distinct states/pages.

You will be given:
1. A screenshot of the current page (use it to understand layout, visual state, and spatial context)
2. The current page URL and title
3. A numbered list of interactive elements on the page
4. A list of elements you have already interacted with on this page
5. (Optional) Information-gain hints showing which element types have been most productive

Use the screenshot to understand the visual context — element positioning, disabled/grayed-out states, modals, overlays, and visual cues that the element list alone may not capture.

Your task: Choose ONE element to interact with that is most likely to lead to a new, unexplored state. Prefer navigation links, tabs, and buttons that suggest new pages/views over elements that would repeat or stay on the same page.

Rules:
- Do NOT choose elements you have already explored (listed in "Already Explored")
- Prefer links and navigation buttons over form inputs (unless filling a form is needed to proceed)
- Avoid destructive actions: do not click "delete", "remove", "logout", "sign out", "reset" buttons
- If an element appears visually disabled or grayed out in the screenshot, skip it
- Consider the information-gain hints: higher scores mean that element type has historically led to new states

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "element_index": <int>,
  "action_type": "click" | "fill" | "select",
  "value": "<string or null>",
  "reasoning": "<brief explanation>"
}\
"""


def _get_system_prompt() -> str:
    return SYSTEM_PROMPT_VISION if VISION_ENABLED else SYSTEM_PROMPT_TEXT

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


def _encode_image(path: str) -> str | None:
    """Base64-encode an image file for vision API calls."""
    try:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None


def _call_llm(system: str, user: str, purpose: str = "action_selection",
              screenshot_path: str | None = None) -> str:
    """Call the LLM and return the response text. Records telemetry.

    If screenshot_path is provided and VISION_ENABLED is True,
    sends the image alongside text for multimodal reasoning.
    """
    provider, client = _get_llm_client()
    start = time.time()
    input_tokens = 0
    output_tokens = 0
    model_used = ""
    error_msg = ""
    success = False
    result = ""

    # Build the user message content (text-only or multimodal)
    use_vision = VISION_ENABLED and screenshot_path
    image_b64 = _encode_image(screenshot_path) if use_vision else None

    try:
        if provider == "anthropic":
            model_used = "claude-sonnet-4-20250514"
            if image_b64:
                user_content = [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": user},
                ]
            else:
                user_content = user
            response = client.messages.create(
                model=model_used,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            result = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        else:
            model_used = os.environ.get("LLM_MODEL", "Kimi-K2.5" if provider == "arc" else "gpt-4o-mini")
            print(f"[INFO] : calling {model_used} from {provider}")

            if image_b64:
                user_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": user},
                ]
            else:
                user_content = user

            response = client.chat.completions.create(
                model=model_used,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )
            result = response.choices[0].message.content
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
            print(f"[INFO] : model responded :{result[:50]}")

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
    screenshot_path: str | None = None,
) -> AgentAction | None:
    """Ask the LLM which element to interact with next.

    Optionally accepts role_rewards — a dict mapping element roles to
    information-gain scores (higher = more likely to discover new states).
    These are appended to the prompt as exploration hints.

    If screenshot_path is provided and VISION_ENABLED, the annotated
    screenshot is sent alongside text for multimodal reasoning.
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

    MAX_RETRIES = 3
    last_error = ""

    for attempt in range(MAX_RETRIES):
        try:
            retry_msg = user_msg
            if last_error:
                retry_msg += f"\n\n[RETRY {attempt}/{MAX_RETRIES}] Your previous response failed to parse: {last_error}\nPlease respond with ONLY valid JSON."

            response = _call_llm(_get_system_prompt(), retry_msg,
                                purpose="action_selection",
                                screenshot_path=screenshot_path)
            data = _parse_json(response)

            # Validate element_index is in range and not already explored
            idx = int(data["element_index"])
            if idx < 0 or idx >= len(elements):
                last_error = f"element_index {idx} is out of range [0, {len(elements)-1}]"
                print(f"  [warn] LLM returned invalid index ({last_error}), retry {attempt+1}/{MAX_RETRIES}")
                continue
            if idx in explored_indices:
                last_error = f"element_index {idx} was already explored"
                print(f"  [warn] LLM picked explored element ({last_error}), retry {attempt+1}/{MAX_RETRIES}")
                continue

            return AgentAction(
                element_index=idx,
                action_type=data.get("action_type", "click"),
                value=data.get("value"),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_error = str(e)
            print(f"  [warn] LLM parse failed ({e}), retry {attempt+1}/{MAX_RETRIES}")
        except Exception as e:
            last_error = str(e)
            print(f"  [warn] LLM call failed ({e}), retry {attempt+1}/{MAX_RETRIES}")

    # All retries exhausted — fallback to info-gain ranked selection
    print(f"  [warn] All {MAX_RETRIES} LLM retries failed, using fallback selection")
    if role_rewards:
        available.sort(key=lambda el: role_rewards.get(el.role, 0.5), reverse=True)
    el = available[0]
    return AgentAction(
        element_index=el.index,
        action_type="click",
        value=None,
        reasoning=f"fallback — LLM failed after {MAX_RETRIES} retries, picked by info-gain score",
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

    MAX_RETRIES = 3
    last_error = ""

    for attempt in range(MAX_RETRIES):
        try:
            retry_msg = user_msg
            if last_error:
                retry_msg += f"\n\n[RETRY {attempt}/{MAX_RETRIES}] Previous response failed: {last_error}\nPlease respond with ONLY a valid JSON array."

            response = _call_llm(LOGIN_SYSTEM_PROMPT, retry_msg, purpose="login")
            data = _parse_json(response)
            if not isinstance(data, list):
                last_error = f"Expected JSON array, got {type(data).__name__}"
                continue

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
            last_error = str(e)
            print(f"  [warn] Login LLM parse failed ({e}), retry {attempt+1}/{MAX_RETRIES}")
        except Exception as e:
            last_error = str(e)
            print(f"  [warn] Login LLM call failed ({e}), retry {attempt+1}/{MAX_RETRIES}")

    print(f"  [warn] Login failed after {MAX_RETRIES} retries")
    return []
