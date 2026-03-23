"""Action execution via Playwright with resilient error handling.

Handles:
- DOM detachment (TargetClosedError, element no longer attached)
- Stale elements that moved or re-rendered
- Navigation timeouts with reload fallback
"""

from __future__ import annotations
from playwright.async_api import Page, Error as PlaywrightError
from .observer import InteractiveElement
from .decider import AgentAction


# Playwright error messages that indicate a detached/stale element
_DETACH_SIGNALS = (
    "target closed",
    "not attached",
    "detached",
    "frame was detached",
    "execution context was destroyed",
    "element is not attached to the DOM",
    "element handle refers to a closed",
)


def _is_detached(error: Exception) -> bool:
    """Check if an error indicates the element was detached from the DOM."""
    msg = str(error).lower()
    return any(signal in msg for signal in _DETACH_SIGNALS)


async def execute_action(
    page: Page,
    action: AgentAction,
    elements: list[InteractiveElement],
) -> tuple[bool, bool]:
    """Execute an action on the page.

    Returns (success, needs_reobserve):
      - success: True if the action executed
      - needs_reobserve: True if the element list is stale and the caller
        should re-extract elements before the next action
    """
    if action.element_index >= len(elements):
        print(f"  [warn] Element index {action.element_index} out of range")
        return False, False

    element = elements[action.element_index]
    selector = element.selector

    # --- Attempt 1: primary selector ---
    try:
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=5000)

        if action.action_type == "click":
            await locator.click(timeout=5000)
        elif action.action_type == "fill":
            await locator.fill(action.value or "", timeout=5000)
        elif action.action_type == "select":
            await locator.select_option(action.value or "", timeout=5000)
        else:
            await locator.click(timeout=5000)

        await _wait_for_settle(page)
        return True, False

    except Exception as e:
        if _is_detached(e):
            print(f"  [warn] Element detached from DOM: '{element.description}' — needs re-observation")
            return False, True  # Signal caller to re-observe

        print(f"  [warn] Primary selector failed on '{element.description}': {e}")

    # --- Attempt 2: text-based fallback ---
    try:
        fallback = page.get_by_text(element.name, exact=False).first
        await fallback.wait_for(state="visible", timeout=3000)

        if action.action_type == "fill":
            await fallback.fill(action.value or "", timeout=3000)
        else:
            await fallback.click(timeout=3000)

        await _wait_for_settle(page)
        return True, False

    except Exception as e:
        if _is_detached(e):
            print(f"  [warn] Fallback also detached — page likely re-rendered")
            return False, True

        print(f"  [warn] Fallback selector also failed: {e}")

    # --- Attempt 3: role-based locator ---
    try:
        role_locator = page.get_by_role(element.role, name=element.name).first
        await role_locator.wait_for(state="visible", timeout=3000)

        if action.action_type == "fill":
            await role_locator.fill(action.value or "", timeout=3000)
        else:
            await role_locator.click(timeout=3000)

        await _wait_for_settle(page)
        return True, False

    except Exception as e:
        if _is_detached(e):
            return False, True
        print(f"  [warn] All 3 selector strategies failed for '{element.description}'")
        return False, False


async def _wait_for_settle(page: Page, timeout: int = 5000) -> None:
    """Wait for page to settle after an action."""
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:
        pass  # Timeout is fine — some actions don't trigger network activity


async def safe_navigate(page: Page, url: str, timeout: int = 15000) -> tuple[bool, str]:
    """Navigate to a URL with timeout handling and reload fallback.

    Returns (success, error_msg).
    If navigation times out, attempts a reload. If that also fails,
    returns False so the caller can mark the edge as a dead end.
    """
    try:
        await page.goto(url, wait_until="networkidle", timeout=timeout)
        return True, ""
    except Exception as e1:
        error1 = str(e1)
        print(f"  [warn] Navigation to {url} failed: {error1}")

        # Attempt reload
        try:
            print(f"  [info] Attempting page reload...")
            await page.reload(wait_until="networkidle", timeout=timeout)
            # Check if we landed on the right page
            if url in page.url or page.url in url:
                return True, ""
            print(f"  [warn] Reload landed on {page.url} instead of {url}")
            return False, f"reload landed on wrong page: {page.url}"
        except Exception as e2:
            error2 = str(e2)
            print(f"  [warn] Reload also failed: {error2}")
            return False, f"navigation failed: {error1}; reload failed: {error2}"
