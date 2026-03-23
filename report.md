# AI Web Agent — Project Report

## What I Built

An **AI-powered web agent** that autonomously explores any web application and produces a **state graph** — a directed graph where nodes are distinct UI states (pages/views) and edges are the user actions (clicks, form fills) that transition between them.

Given a starting URL, the agent launches a headless browser, discovers interactive elements, uses an LLM to decide what to click next, and records every new state it finds — building a complete map of the application's navigation structure.

---

## How It Works

The agent runs a simple but effective loop: **Observe → Decide → Act → Record**.

### Step 1: Observe

The agent extracts interactive elements from the current page using JavaScript evaluation on the live DOM. It doesn't use raw HTML (too noisy, too many tokens). Instead, it builds a compact list of actionable elements — buttons, links, inputs, tabs — each with a role, name, and CSS selector.

To avoid overwhelming the LLM, elements are **tiered**:
- **Tier 1** (always included): Buttons, inputs, tabs, and links inside navigation/header areas — these are the elements most likely to lead to new pages.
- **Tier 2** (fills remaining slots): Content links from the page body.
- **Hard cap: 40 elements.** This keeps LLM prompts focused. Without this cap, pages like IANA's Protocol Registries would send 4000+ links to the LLM.

### Step 2: Decide

The element list, page URL, title, already-explored elements, and information-gain hints are sent to an LLM (Kimi-K2.5 via VT ARC API, or Anthropic/OpenAI). The LLM returns a JSON response picking one element to interact with and explaining why.

If the LLM returns bad JSON or picks an invalid element, the agent **retries up to 3 times**, appending the parse error to the prompt so the LLM can self-correct. If all retries fail, the agent falls back to a **deterministic selection** based on information-gain scores — so it always makes progress.

### Step 3: Act

The agent executes the chosen action via Playwright. It uses a **3-tier selector fallback**:
1. Primary CSS/role selector from element extraction
2. Text-based matching (`get_by_text`)
3. Role-based matching (`get_by_role`)

If the element was detached from the DOM (common in SPAs), the agent detects this and re-observes the page instead of crashing.

### Step 4: Record

After acting, the agent fingerprints the resulting page using **two tiers of state matching**:

1. **Structural DOM Fingerprint** — SHA-256 hash of the page's URL path + title + accessibility tree structure. This catches most revisits.
2. **Perceptual Hash (pHash)** — A visual hash of the screenshot. Two pages with pHash distance ≤ 8 are considered the same state. This catches edge cases like A/B test variants and cookie banners that change the DOM but not the visual appearance.

If the fingerprint matches an existing state, the agent knows it's revisiting. If not, it records a new node in the graph, takes a screenshot, and adds an edge from the previous state.

---

## What Makes the Exploration Smart

### Information-Gain Learning

The agent doesn't just randomly click things. It tracks which **types** of elements (buttons, links, tabs, etc.) have historically led to discovering new states during the current run.

Each role gets a reward score updated via exponential moving average:
- Clicked a tab and found a new state? Tab score goes up.
- Clicked a pagination link and landed on an existing state? Link score goes down.

These scores are passed to the LLM as hints ("tabs have been very productive, links less so") and also serve as the fallback ranking when the LLM fails. This approximates a **contextual bandit** — the agent gets better at exploring as the run progresses.

### Backtracking

When a state has no unexplored elements left, the agent backtracks. It iterates through all recorded states, finds one with unexplored elements, and navigates directly to its URL. If a URL is unreachable (site changed, session expired), the state is marked as a dead end so the agent stops wasting steps on it.

---

## Error Resilience

The agent is designed to degrade gracefully:

| Problem | How It's Handled |
|---------|-----------------|
| LLM returns invalid JSON | Retry up to 3 times with error feedback appended to prompt |
| LLM picks already-explored element | Retry with the constraint re-emphasized |
| All LLM retries fail | Fall back to info-gain ranked selection (no LLM needed) |
| DOM element detached mid-click | Detect via error pattern matching, re-observe page |
| Navigation times out | Attempt page reload; if that fails, mark state as dead end |
| Selector doesn't match | Try text-based fallback, then role-based fallback |

---

## Vision Mode

With the `--vision` flag, the agent sends **screenshots alongside text** to a Vision Language Model. The screenshot is base64-encoded and included in the LLM prompt using the provider's multimodal API format.

A vision-specific system prompt tells the LLM to use the screenshot for spatial reasoning — understanding element layout, detecting visually disabled elements, recognizing modals and overlays that the text-only element list might miss.

This is optional because it costs more tokens and adds latency. The text-only mode is sufficient for most sites.

---

## LLM Integration

The agent supports three LLM providers:

| Provider | API Key Env Var | Default Model | Notes |
|----------|----------------|---------------|-------|
| VT ARC (primary) | `ARC_API_KEY` | Kimi-K2.5 | On-prem at Virginia Tech, OpenAI-compatible API |
| Anthropic | `ANTHROPIC_API_KEY` | Claude Sonnet | Native Anthropic SDK |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | Standard OpenAI SDK |

The LLM is used for exactly two things:
1. **Action selection** — picking which element to interact with next
2. **Login form detection** — identifying username/password fields and submit button

Everything else (element extraction, fingerprinting, execution, graph management) is deterministic code.

---

## Output

Each run produces three artifacts:

1. **`graph.json`** — The state graph with all nodes (states) and edges (actions)
2. **`telemetry.json`** — Per-LLM-call metrics: provider, model, tokens, latency, success/failure
3. **`screenshots/`** — PNG screenshot for every discovered state

Using `--name <run_name>`, each run gets its own directory: `output/<name>_<max_steps>/`

---

## Project Structure

```
testsprite/
├── agent.py               # Main entry — agent loop, CLI, info-gain learning
├── core/
│   ├── browser.py         # Playwright browser management + accessibility tree
│   ├── observer.py        # Tiered element extraction + page context formatting
│   ├── decider.py         # LLM calls — prompts, retries, vision, telemetry
│   ├── executor.py        # Action execution — selector fallback, DOM detach handling
│   ├── state.py           # State fingerprinting — structural hash + pHash
│   ├── graph.py           # Graph model — nodes, edges, multi-tier matching
│   └── som.py             # Set-of-Mark annotation (scaffolded for future use)
├── DESIGN.md              # Architecture design document
├── report.md              # This file
└── output/                # Per-run output directories
```

---

## Key Design Decisions

1. **Accessibility tree over raw HTML** — 10-50x smaller, no CSS/script noise, maps to user-perceivable elements.
2. **Tiered element cap at 40** — Prevents token waste on link-heavy pages. Navigation elements get priority.
3. **Multi-tier state identity** — URLs alone are unreliable for SPAs. Full DOM comparison is too strict. Two tiers (hash + pHash) balance speed and robustness.
4. **LLM retry with error feedback** — LLMs often self-correct when shown their mistake. Three retries with the error appended is cheap and effective.
5. **Deterministic fallback** — The agent never stalls. If the LLM fails entirely, info-gain scores provide a reasonable heuristic for choosing the next action.
6. **Surgical LLM use** — Only two LLM call sites in the entire codebase (action selection + login). Everything else is deterministic. This keeps costs low and behavior predictable.
