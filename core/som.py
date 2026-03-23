"""Set-of-Mark (SoM) prompting — annotate screenshots with numbered bounding boxes.

This module takes a screenshot and overlays numbered bounding boxes on each
interactive element, producing an annotated image that can be sent to a
Vision-Language Model (VLM) for more spatially-aware action selection.

Reference: Yang et al., "Set-of-Mark Prompting Unleashes Extraordinary Visual
Grounding in GPT-4V" (2023).
"""

from __future__ import annotations
from dataclasses import dataclass
from playwright.async_api import Page

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class ElementBBox:
    index: int
    role: str
    name: str
    x: float
    y: float
    width: float
    height: float


async def extract_element_bboxes(page: Page) -> list[ElementBBox]:
    """Extract bounding boxes for all interactive elements on the page."""
    raw = await page.evaluate('''() => {
        const results = [];
        const seen = new Set();
        const selectors = [
            'a[href]', 'button', 'input', 'select', 'textarea',
            '[role="button"]', '[role="link"]', '[role="tab"]',
            '[role="menuitem"]', '[role="checkbox"]', '[role="radio"]',
            '[tabindex="0"]',
        ];
        for (const sel of selectors) {
            for (const el of document.querySelectorAll(sel)) {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') ||
                    ({a:'link', button:'button', input:'textbox', select:'combobox', textarea:'textbox'}[tag] || tag);
                const name = el.getAttribute('aria-label') || el.textContent?.trim().slice(0,50) || '';
                if (!name) continue;

                const key = `${role}:${name}`;
                if (seen.has(key)) continue;
                seen.add(key);

                results.push({
                    role, name,
                    x: rect.x, y: rect.y,
                    width: rect.width, height: rect.height,
                });
            }
        }
        return results;
    }''')

    return [
        ElementBBox(index=i, role=r["role"], name=r["name"],
                    x=r["x"], y=r["y"], width=r["width"], height=r["height"])
        for i, r in enumerate(raw)
    ]


def annotate_screenshot(
    screenshot_path: str,
    bboxes: list[ElementBBox],
    output_path: str,
) -> str:
    """Draw numbered bounding boxes on a screenshot.

    Returns the path to the annotated image.
    """
    if not PIL_AVAILABLE:
        print("  [warn] PIL not available, skipping SoM annotation")
        return screenshot_path

    img = Image.open(screenshot_path)
    draw = ImageDraw.Draw(img)

    # Color palette for different roles
    role_colors = {
        "link": (0, 120, 215),       # blue
        "button": (220, 50, 50),     # red
        "tab": (50, 180, 50),        # green
        "textbox": (180, 120, 0),    # orange
        "combobox": (140, 0, 200),   # purple
        "menuitem": (0, 180, 180),   # teal
    }
    default_color = (100, 100, 100)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for bbox in bboxes:
        color = role_colors.get(bbox.role, default_color)
        x1, y1 = bbox.x, bbox.y
        x2, y2 = x1 + bbox.width, y1 + bbox.height

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background
        label = str(bbox.index)
        label_w, label_h = 20, 18
        draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=color)
        draw.text((x1 + 3, y1 - label_h + 1), label, fill="white", font=font)

    img.save(output_path)
    return output_path
