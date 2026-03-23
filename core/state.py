"""State fingerprinting and identity tracking.

Multi-tier state identity:
  Tier 1: Structural DOM hash (fast, exact match)
  Tier 2: Perceptual hash of screenshot (visual similarity, robust to minor DOM changes)
  Tier 3: LLM judge for ambiguous cases (functional equivalence check)
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

try:
    import imagehash
    from PIL import Image
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False


@dataclass
class StateNode:
    id: str
    url: str
    title: str
    dom_fingerprint: str
    phash: str                          # perceptual hash of screenshot
    interactive_elements: list[str]
    screenshot: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "dom_fingerprint": self.dom_fingerprint,
            "phash": self.phash,
            "interactive_elements": self.interactive_elements,
            "screenshot": self.screenshot,
            "timestamp": self.timestamp,
        }


@dataclass
class ActionEdge:
    from_state: str
    to_state: str
    action_type: str
    target: str
    selector: str
    description: str

    def to_dict(self) -> dict:
        return {
            "from": self.from_state,
            "to": self.to_state,
            "action": {
                "type": self.action_type,
                "target": self.target,
                "selector": self.selector,
                "description": self.description,
            },
        }


# ---------------------------------------------------------------------------
# Tier 1: Structural DOM fingerprint
# ---------------------------------------------------------------------------

def _extract_structure(tree: dict | list | None, depth: int = 0) -> str:
    """Extract structural skeleton from accessibility tree, ignoring dynamic text values."""
    if tree is None:
        return ""
    if isinstance(tree, list):
        return "".join(_extract_structure(child, depth) for child in tree)

    role = tree.get("role", "")
    name = tree.get("name", "")
    # Keep name only for structural elements; truncate to avoid data-dependent hashing
    structural_roles = {"heading", "link", "button", "tab", "menuitem", "navigation", "banner", "main", "complementary"}
    if role in structural_roles and name:
        label = name[:50]
    else:
        label = ""

    parts = [f"{depth}:{role}:{label}"]
    for child in (tree.get("children") or []):
        parts.append(_extract_structure(child, depth + 1))
    return "|".join(parts)


def compute_fingerprint(url: str, title: str, accessibility_tree: dict | None) -> str:
    """Compute a structural state fingerprint from URL path, title, and DOM structure."""
    parsed = urlparse(url)
    url_path = parsed.path or "/"
    structure = _extract_structure(accessibility_tree)
    raw = f"{url_path}||{title}||{structure}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tier 2: Perceptual hash of screenshot
# ---------------------------------------------------------------------------

def compute_phash(screenshot_path: str) -> str:
    """Compute perceptual hash of a screenshot image.

    Returns a hex string. Similar images produce hashes with small Hamming distance.
    Returns empty string if imagehash is not available or image can't be loaded.
    """
    if not PHASH_AVAILABLE:
        return ""
    try:
        img = Image.open(screenshot_path)
        return str(imagehash.phash(img))
    except Exception:
        return ""


def phash_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two perceptual hashes.

    Lower distance = more visually similar. Returns -1 if comparison not possible.
    """
    if not PHASH_AVAILABLE or not hash_a or not hash_b:
        return -1
    try:
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except Exception:
        return -1


# Threshold: pages with pHash distance <= this are considered visually identical
PHASH_SIMILARITY_THRESHOLD = 8
