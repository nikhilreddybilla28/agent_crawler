"""Action execution via Playwright."""

from __future__ import annotations
from playwright.async_api import Page
from .observer import InteractiveElement
from .decider import AgentAction


async def execute_action(
    page: Page,
    action: AgentAction,
    elements: list[InteractiveElement],
) -> bool:
    """Execute an action on the page. Returns True if successful."""
    if action.element_index >= len(elements):
        print(f"  [warn] Element index {action.element_index} out of range")
        return False

    element = elements[action.element_index]
    selector = element.selector

    try:
        # Wait for the element to be visible
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

        # Wait for any navigation or dynamic content to settle
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass  # Timeout is fine — some actions don't trigger network activity

        return True

    except Exception as e:
        print(f"  [warn] Action failed on '{element.description}': {e}")
        # Try a fallback approach: text-based selector
        try:
            fallback = page.get_by_text(element.name, exact=False).first
            if action.action_type == "fill":
                await fallback.fill(action.value or "", timeout=3000)
            else:
                await fallback.click(timeout=3000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            return True
        except Exception:
            return False
