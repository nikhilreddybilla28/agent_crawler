"""Page observation — extract interactive elements and page context from the browser."""

from __future__ import annotations
from dataclasses import dataclass
from playwright.async_api import Page


INTERACTIVE_ROLES = {
    "link", "button", "menuitem", "tab", "checkbox", "radio",
    "textbox", "combobox", "searchbox", "switch", "option",
    "spinbutton", "slider",
}


@dataclass
class InteractiveElement:
    index: int
    role: str
    name: str
    selector: str
    description: str


def _walk_tree(node: dict | None, elements: list[InteractiveElement], path: str = "") -> None:
    """Recursively walk accessibility tree and collect interactive elements."""
    if node is None:
        return

    role = node.get("role", "")
    name = node.get("name", "")

    if role in INTERACTIVE_ROLES and name:
        idx = len(elements)
        selector = _build_selector(role, name)
        desc = f"{role}: {name}"
        elements.append(InteractiveElement(
            index=idx,
            role=role,
            name=name,
            selector=selector,
            description=desc,
        ))

    for child in (node.get("children") or []):
        _walk_tree(child, elements, path)


def _build_selector(role: str, name: str) -> str:
    """Build a Playwright-compatible selector from role and name."""
    escaped = name.replace('"', '\\"')
    return f'role={role}[name="{escaped}"]'


def extract_interactive_elements(accessibility_tree: dict | None) -> list[InteractiveElement]:
    """Extract all interactive elements from an accessibility tree."""
    elements: list[InteractiveElement] = []
    _walk_tree(accessibility_tree, elements)
    return elements


async def extract_elements_via_page(page: Page) -> list[InteractiveElement]:
    """Extract interactive elements directly from the page using JS evaluation.

    This is a more reliable fallback that finds clickable/interactive elements
    directly in the DOM.
    """
    raw = await page.evaluate('''() => {
        const tier1 = [];  // Always include: buttons, inputs, tabs + nav/header links
        const tier2 = [];  // Fill remaining slots: content links from <main>
        const seen = new Set();

        const selectors = [
            'a[href]', 'button', 'input', 'select', 'textarea',
            '[role="button"]', '[role="link"]', '[role="tab"]',
            '[role="menuitem"]', '[role="checkbox"]', '[role="radio"]',
            '[role="switch"]', '[onclick]', '[tabindex="0"]',
        ];

        function isInsideNav(el) {
            let p = el.parentElement;
            while (p) {
                const t = p.tagName?.toLowerCase();
                const r = p.getAttribute('role');
                if (t === 'nav' || t === 'header' || t === 'aside' ||
                    r === 'navigation' || r === 'banner' || r === 'complementary' ||
                    r === 'menu' || r === 'menubar' || r === 'tablist') return true;
                p = p.parentElement;
            }
            return false;
        }

        for (const sel of selectors) {
            for (const el of document.querySelectorAll(sel)) {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') ||
                    ({a:'link', button:'button', input:'textbox', select:'combobox', textarea:'textbox'}[tag] || tag);
                const name = el.getAttribute('aria-label')
                    || el.getAttribute('title')
                    || el.getAttribute('placeholder')
                    || el.textContent?.trim().slice(0, 80)
                    || el.getAttribute('name')
                    || '';

                if (!name) continue;

                const key = `${role}:${name}`;
                if (seen.has(key)) continue;
                seen.add(key);

                let cssSelector = '';
                if (el.id) cssSelector = `#${CSS.escape(el.id)}`;
                else if (el.getAttribute('data-testid'))
                    cssSelector = `[data-testid="${el.getAttribute('data-testid')}"]`;

                let entry;
                if (el.type === 'checkbox') entry = {role: 'checkbox', name, cssSelector};
                else if (el.type === 'radio') entry = {role: 'radio', name, cssSelector};
                else if (el.type === 'password') entry = {role: 'textbox', name: name || 'Password', cssSelector};
                else entry = {role, name, cssSelector};

                // Tier 1: all non-link roles + links inside nav/header
                if (role !== 'link' || isInsideNav(el)) tier1.push(entry);
                else tier2.push(entry);
            }
        }
        return {tier1, tier2};
    }''')

    MAX_ELEMENTS = 40

    tier1 = raw.get("tier1", [])
    tier2 = raw.get("tier2", [])

    # Tier 1 always included (up to cap), Tier 2 fills remaining slots
    combined = tier1[:MAX_ELEMENTS]
    remaining = MAX_ELEMENTS - len(combined)
    if remaining > 0:
        combined.extend(tier2[:remaining])

    skipped = len(tier1) + len(tier2) - len(combined)
    if skipped > 0:
        print(f"  [info] {len(tier1)} nav elements, {len(tier2)} content links — capped to {len(combined)}, skipped {skipped}")

    elements: list[InteractiveElement] = []
    for item in combined:
        idx = len(elements)
        role = item["role"]
        name = item["name"]
        css = item.get("cssSelector", "")
        selector = css if css else _build_selector(role, name)
        elements.append(InteractiveElement(
            index=idx,
            role=role,
            name=name,
            selector=selector,
            description=f"{role}: {name}",
        ))
    return elements


def format_page_context(url: str, title: str, elements: list[InteractiveElement]) -> str:
    """Format page context for the LLM."""
    lines = [
        f"Current URL: {url}",
        f"Page Title: {title}",
        "",
        "Interactive Elements:",
    ]
    if not elements:
        lines.append("  (no interactive elements found)")
    else:
        for el in elements:
            lines.append(f"  [{el.index}] {el.description}")
    return "\n".join(lines)
