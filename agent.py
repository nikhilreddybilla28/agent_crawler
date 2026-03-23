#!/usr/bin/env python3
"""
AI Web Agent — Interactive Site Graph Mapper

Autonomously explores a web application and produces a state graph:
a directed graph where nodes are distinct UI states and edges are
the actions taken to transition between them.

Usage:
    python agent.py --url https://example.com [--max-steps 20] [--username USER --password PASS]

Requires: ARC_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY environment variable.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.browser import BrowserManager
from core.observer import extract_interactive_elements, extract_elements_via_page, format_page_context
from core.decider import decide_action, decide_login_actions, get_telemetry
import core.decider as decider_module
from core.executor import execute_action, safe_navigate
from core.state import compute_fingerprint, compute_phash, StateNode, ActionEdge
from core.graph import StateGraph


class WebAgent:
    def __init__(
        self,
        start_url: str,
        max_steps: int = 20,
        username: str | None = None,
        password: str | None = None,
        headless: bool = True,
        vision: bool = False,
        screenshot_dir: str = "screenshots",
        output_path: str = "output/graph.json",
    ):
        self.start_url = start_url
        self.max_steps = max_steps
        self.username = username
        self.password = password
        self.headless = headless
        self.vision = vision
        self.screenshot_dir = screenshot_dir
        self.output_path = output_path

        # Enable vision mode in the decider module
        decider_module.VISION_ENABLED = vision

        self.browser = BrowserManager()
        self.graph = StateGraph()
        # Track explored elements per state: {state_id: set(element_indices)}
        self.explored: dict[str, set[int]] = {}
        # Navigation stack for backtracking
        self.nav_stack: list[str] = []
        # Information-gain scores per element role (learned during exploration)
        self._role_rewards: dict[str, float] = {
            "link": 1.0, "button": 1.5, "tab": 2.0, "menuitem": 1.8,
            "textbox": 0.3, "combobox": 0.5, "checkbox": 0.3,
        }
        self._role_attempts: dict[str, int] = {}

    async def _get_elements(self):
        """Extract interactive elements — tries page-based extraction first, falls back to tree."""
        try:
            elements = await extract_elements_via_page(self.browser.page)
            if elements:
                return elements
        except Exception:
            pass
        tree = await self.browser.get_accessibility_tree()
        return extract_interactive_elements(tree)

    async def run(self) -> StateGraph:
        """Main agent loop."""
        print(f"Starting exploration of {self.start_url}")
        print(f"Max steps: {self.max_steps}\n")

        page = await self.browser.start(headless=self.headless)
        try:
            ok, err = await safe_navigate(self.browser.page, self.start_url)
            if not ok:
                print(f"FATAL: Could not load start URL: {err}")
                return self.graph

            # Handle login if credentials provided
            if self.username and self.password:
                await self._handle_login()

            # Record initial state
            current_state_id = await self._observe_and_record()
            if current_state_id:
                self.nav_stack.append(current_state_id)

            # Main exploration loop
            step = 0
            while step < self.max_steps:
                step += 1
                print(f"\n--- Step {step}/{self.max_steps} ---")

                result = await self._exploration_step()
                if result == "done":
                    print("Exploration complete — no more unexplored elements.")
                    break
                elif result == "backtrack":
                    backtracked = await self._backtrack()
                    if not backtracked:
                        print("No states with unexplored elements remain. Done.")
                        break

            print(f"\n{'='*50}")
            print(self.graph.summary())
            self.graph.save(self.output_path)

            # Save telemetry alongside graph
            telemetry = get_telemetry()
            telemetry_path = self.output_path.replace(".json", "_telemetry.json")
            Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)
            with open(telemetry_path, "w") as f:
                json.dump(telemetry, f, indent=2)
            print(f"Telemetry saved to {telemetry_path}")
            print(f"  LLM calls: {telemetry['total_calls']}, "
                  f"tokens: {telemetry['total_tokens']}, "
                  f"success rate: {telemetry['success_rate']:.0%}")

            return self.graph

        finally:
            await self.browser.close()

    async def _handle_login(self) -> None:
        """Attempt to log in using provided credentials."""
        print("Attempting login...")
        elements = await self._get_elements()
        url = await self.browser.get_url()
        title = await self.browser.get_title()
        context = format_page_context(url, title, elements)

        actions = decide_login_actions(context, self.username, self.password)
        for action in actions:
            success = await execute_action(self.browser.page, action, elements)
            if success:
                print(f"  Login action: {action.action_type} on [{action.element_index}]")
            await asyncio.sleep(0.5)

        # Wait for post-login navigation
        try:
            await self.browser.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        print("Login sequence completed.\n")

    async def _observe_and_record(self) -> str | None:
        """Observe the current page and record it as a state. Returns state_id.

        Uses multi-tier state matching:
          Tier 1: exact structural DOM fingerprint
          Tier 2: perceptual hash similarity (catches minor DOM diffs)
        """
        url = await self.browser.get_url()
        title = await self.browser.get_title()
        tree = await self.browser.get_accessibility_tree()
        elements = await self._get_elements()

        fingerprint = compute_fingerprint(url, title, tree)

        # Take screenshot early — needed for Tier 2 pHash comparison
        temp_screenshot = f"{self.screenshot_dir}/_temp.png"
        try:
            await self.browser.screenshot(temp_screenshot)
        except Exception as e:
            print(f"  [warn] Screenshot failed: {e}")
            temp_screenshot = ""

        phash = compute_phash(temp_screenshot) if temp_screenshot else ""

        # Multi-tier matching
        existing_id = self.graph.has_fingerprint(fingerprint, phash)
        if existing_id:
            print(f"  Recognized existing state: {existing_id} ({title})")
            # Clean up temp screenshot
            if temp_screenshot:
                Path(temp_screenshot).unlink(missing_ok=True)
            return existing_id

        # New state — keep the screenshot with proper name
        state_id = self.graph.next_state_id()
        screenshot_path = f"{self.screenshot_dir}/{state_id}.png"
        if temp_screenshot:
            Path(temp_screenshot).rename(screenshot_path)
        else:
            screenshot_path = ""

        node = StateNode(
            id=state_id,
            url=url,
            title=title,
            dom_fingerprint=fingerprint,
            phash=phash,
            interactive_elements=[el.description for el in elements],
            screenshot=screenshot_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.graph.add_node(node)
        self.explored[state_id] = set()

        print(f"  NEW state: {state_id} — {title} ({url})")
        print(f"  Found {len(elements)} interactive elements, phash={phash[:8]}..." if phash else f"  Found {len(elements)} interactive elements")
        return state_id

    async def _exploration_step(self) -> str:
        """Execute one exploration step. Returns 'ok', 'backtrack', or 'done'."""
        url = await self.browser.get_url()
        title = await self.browser.get_title()
        tree = await self.browser.get_accessibility_tree()
        elements = await self._get_elements()

        # Identify current state
        fingerprint = compute_fingerprint(url, title, tree)
        current_id = self.graph.has_fingerprint(fingerprint)
        if not current_id:
            current_id = await self._observe_and_record()
            if current_id:
                self.nav_stack.append(current_id)

        if not current_id:
            return "backtrack"

        # Get explored set for this state
        explored_here = self.explored.get(current_id, set())
        context = format_page_context(url, title, elements)

        # Get screenshot path for vision mode
        current_screenshot = None
        if self.vision and current_id and current_id in self.graph.nodes:
            current_screenshot = self.graph.nodes[current_id].screenshot

        # Ask LLM what to do (with information-gain hints + optional screenshot)
        action = decide_action(context, explored_here, elements, self._role_rewards,
                               screenshot_path=current_screenshot)
        if action is None:
            print(f"  No unexplored elements in {current_id}")
            return "backtrack"

        # Mark element as explored
        explored_here.add(action.element_index)
        self.explored[current_id] = explored_here

        target_desc = (
            elements[action.element_index].description
            if action.element_index < len(elements)
            else f"element_{action.element_index}"
        )
        print(f"  Action: {action.action_type} on [{action.element_index}] {target_desc}")
        print(f"  Reason: {action.reasoning}")

        # Track the role for information-gain learning
        acted_role = elements[action.element_index].role if action.element_index < len(elements) else ""
        pre_state_count = len(self.graph.nodes)

        # Execute the action
        success, needs_reobserve = await execute_action(self.browser.page, action, elements)

        if needs_reobserve:
            print("  DOM detached — re-observing page state")
            # Element list is stale; re-extract and let the next step pick up
            self._update_role_reward(acted_role, discovered_new=False)
            return "ok"

        if not success:
            print("  Action failed — skipping")
            self._update_role_reward(acted_role, discovered_new=False)
            return "ok"

        await asyncio.sleep(1)

        # Observe the result
        new_state_id = await self._observe_and_record()

        # Information-gain feedback: did this action discover a new state?
        discovered_new = len(self.graph.nodes) > pre_state_count
        self._update_role_reward(acted_role, discovered_new)

        if new_state_id and new_state_id != current_id:
            self.nav_stack.append(new_state_id)

        # Record the edge
        if new_state_id:
            target_el = elements[action.element_index] if action.element_index < len(elements) else None
            edge = ActionEdge(
                from_state=current_id,
                to_state=new_state_id,
                action_type=action.action_type,
                target=target_el.name if target_el else "",
                selector=target_el.selector if target_el else "",
                description=f"{action.action_type} '{target_el.name if target_el else 'unknown'}'",
            )
            self.graph.add_edge(edge)

        return "ok"

    def _update_role_reward(self, role: str, discovered_new: bool) -> None:
        """Update information-gain reward for a role using exponential moving average."""
        if not role:
            return
        reward = 1.0 if discovered_new else 0.0
        alpha = 0.3  # learning rate
        current = self._role_rewards.get(role, 1.0)
        self._role_rewards[role] = current * (1 - alpha) + reward * alpha
        self._role_attempts[role] = self._role_attempts.get(role, 0) + 1

    async def _backtrack(self) -> bool:
        """Navigate back to a state with unexplored elements.

        Uses safe_navigate with timeout/reload fallback. If a state's URL
        is unreachable, marks all its elements as explored (dead end) so
        we don't keep trying to backtrack to it.
        """
        for state_id, node in self.graph.nodes.items():
            explored_here = self.explored.get(state_id, set())
            total_elements = len(node.interactive_elements)
            if len(explored_here) < total_elements:
                print(f"  Backtracking to {state_id} ({node.url})")
                ok, err = await safe_navigate(self.browser.page, node.url)
                if ok:
                    await asyncio.sleep(1)
                    self.nav_stack.append(state_id)
                    return True
                else:
                    # Mark this state as a dead end — can't reach it anymore
                    print(f"  [dead-end] Marking {state_id} as fully explored (unreachable: {err})")
                    self.explored[state_id] = set(range(total_elements))

        if len(self.nav_stack) > 1:
            self.nav_stack.pop()
            print("  Backtracking via browser history")
            await self.browser.go_back()
            await asyncio.sleep(1)
            return True

        return False


def main():
    parser = argparse.ArgumentParser(
        description="AI Web Agent — Interactive Site Graph Mapper"
    )
    parser.add_argument("--url", required=True, help="Starting URL to explore")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum exploration steps (default: 20)")
    parser.add_argument("--username", default=None, help="Login username (optional)")
    parser.add_argument("--password", default=None, help="Login password (optional)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser in headless mode")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show browser window")
    parser.add_argument("--vision", action="store_true", default=False, help="Send screenshots to VLM alongside text (uses Kimi-K2.5 vision)")
    parser.add_argument("--name", default=None, help="Run name — creates output/<name>/ with graph.json, telemetry, and screenshots")
    parser.add_argument("--output", default=None, help="Output JSON path (overrides --name)")
    args = parser.parse_args()

    # Resolve output path and screenshot dir
    if args.output:
        output_path = args.output
        screenshot_dir = "screenshots"
    elif args.name:
        run_dir = f"output/{args.name}_{args.max_steps}"
        output_path = f"{run_dir}/graph.json"
        screenshot_dir = f"{run_dir}/screenshots"
    else:
        output_path = "output/graph.json"
        screenshot_dir = "screenshots"

    if args.vision:
        print("[vision] Screenshot mode enabled — sending images to VLM")

    agent = WebAgent(
        start_url=args.url,
        max_steps=args.max_steps,
        username=args.username,
        password=args.password,
        headless=args.headless,
        vision=args.vision,
        screenshot_dir=screenshot_dir,
        output_path=output_path,
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
