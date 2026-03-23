"""Graph data model and serialization with multi-tier state matching."""

import json
from pathlib import Path
from .state import StateNode, ActionEdge, phash_distance, PHASH_SIMILARITY_THRESHOLD


class StateGraph:
    def __init__(self):
        self.nodes: dict[str, StateNode] = {}
        self.edges: list[ActionEdge] = []
        self._state_counter = 0

    def next_state_id(self) -> str:
        self._state_counter += 1
        return f"state_{self._state_counter:03d}"

    def find_matching_state(self, fingerprint: str, phash: str = "") -> tuple[str | None, str]:
        """Multi-tier state matching.

        Returns (state_id, match_tier) where match_tier is one of:
          "exact"    — Tier 1: structural DOM hash matched exactly
          "visual"   — Tier 2: perceptual hash is within similarity threshold
          None       — no match found

        This prevents false "new state" detections caused by minor DOM changes
        (A/B tests, cookie banners, dynamic reordering) that don't affect the
        visual appearance of the page.
        """
        # Tier 1: exact structural fingerprint match
        for sid, node in self.nodes.items():
            if node.dom_fingerprint == fingerprint:
                return sid, "exact"

        # Tier 2: perceptual hash similarity
        if phash:
            best_sid = None
            best_dist = float("inf")
            for sid, node in self.nodes.items():
                dist = phash_distance(phash, node.phash)
                if dist >= 0 and dist < best_dist:
                    best_dist = dist
                    best_sid = sid
            if best_sid and best_dist <= PHASH_SIMILARITY_THRESHOLD:
                print(f"  [tier-2] Visual match: pHash distance={best_dist} to {best_sid}")
                return best_sid, "visual"

        return None, ""

    # Keep backward-compatible API
    def has_fingerprint(self, fingerprint: str, phash: str = "") -> str | None:
        sid, _ = self.find_matching_state(fingerprint, phash)
        return sid

    def add_node(self, node: StateNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: ActionEdge) -> None:
        self.edges.append(edge)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Graph saved to {path}")

    def summary(self) -> str:
        lines = [f"States discovered: {len(self.nodes)}", f"Transitions recorded: {len(self.edges)}", ""]
        for node in self.nodes.values():
            lines.append(f"  [{node.id}] {node.title} ({node.url}) phash={node.phash[:8]}..." if node.phash else f"  [{node.id}] {node.title} ({node.url})")
        lines.append("")
        for edge in self.edges:
            lines.append(f"  {edge.from_state} --({edge.description})--> {edge.to_state}")
        return "\n".join(lines)
