# AI Web Agent — Design Document
## Interactive Site Graph Mapper

### 1. Architecture

The agent follows an **observe → decide → act → record** loop:

```
┌─────────────────────────────────────────────────────┐
│                    Agent Loop                        │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │ OBSERVE  │──>│  DECIDE  │──>│   ACT    │         │
│  │(snapshot │   │ (LLM     │   │(Playwright│        │
│  │ page +   │   │  picks   │   │ executes │         │
│  │ pHash)   │   │  action  │   │ action)  │         │
│  └──────────┘   │  + info  │   └────┬─────┘         │
│       ▲         │  gain)   │        │               │
│       │         └──────────┘        │               │
│       │         ┌──────────┐        │               │
│       └─────────│  RECORD  │<───────┘               │
│                 │(multi-tier                         │
│                 │ fingerprint                        │
│                 │ + graph +                          │
│                 │ telemetry)│                        │
│                 └──────────┘                         │
└─────────────────────────────────────────────────────┘
```

**LLM vs. Deterministic Code Split:**

| Component | Approach | Why |
|---|---|---|
| Page observation (DOM extraction, element bboxes) | Deterministic | Reliable, fast, no token cost |
| State fingerprinting (structural hash + pHash) | Deterministic | Consistent, reproducible, multi-tier |
| Action selection ("what to click next") | **LLM + info-gain heuristic** | Semantic understanding + learned reward signals |
| Action execution (click, type, navigate) | Deterministic | Playwright handles this reliably |
| State comparison (new vs. visited) | **Multi-tier** | Tier 1: exact hash, Tier 2: pHash visual similarity |
| Login handling | **LLM** | Needs to understand varied login form layouts |
| Set-of-Mark annotation | Deterministic | Computational overlay for VLM input |

The LLM is used surgically — only where semantic understanding of the page is needed. Everything else is deterministic for reliability and speed.

### 2. Page Representation

The page is represented to the LLM as a **structured accessibility snapshot** with three components:

1. **Accessibility Tree (primary):** A JavaScript-extracted semantic tree of the page — roles, names, values, states. This is compact, structured, and captures what a user would perceive (buttons, links, inputs, headings, etc.) without the noise of raw HTML.

2. **Metadata:** URL, page title — gives the LLM orientation context.

3. **Interactive Elements List:** A filtered, indexed list of actionable elements extracted via direct DOM queries (buttons, links, inputs, selects, checkboxes, etc.), each with a stable selector. This gives the LLM a clear "menu" of possible actions.

4. **Information-Gain Hints (new):** Role-based reward scores learned during exploration, indicating which element types have historically led to new state discoveries.

**Why not raw HTML?** Too noisy, too many tokens, contains layout/style info irrelevant to action decisions.

**Set-of-Mark (SoM) Mode (optional):** For VLM-capable models, we support annotating screenshots with numbered bounding boxes overlaid on each interactive element (`core/som.py`). This merges the reliability of deterministic element extraction with the spatial reasoning of vision models. The annotated screenshot can be sent alongside the element list to reduce hallucinated actions on visually complex pages.

**Tradeoff:** SoM requires a VLM endpoint (higher latency/cost). The default text-only mode is faster and sufficient for most sites.

### 3. State Identity

States are identified using a **multi-tier matching system** that balances speed with robustness:

#### Tier 1: Structural DOM Fingerprint (fast, exact)
```
fingerprint = sha256(url_path + title + structural_tree_hash)
```
The structural tree hash captures the "shape" of the page — element roles and names — while stripping dynamic text content (timestamps, counters) to avoid false negatives.

#### Tier 2: Perceptual Hash (visual similarity)
A perceptual hash (pHash) of the screenshot captures what the page *looks like*. Two pages with pHash Hamming distance ≤ 8 are considered visually identical, even if their DOM structures differ slightly.

This catches false "new state" detections caused by:
- A/B tests injecting minor DOM changes
- Cookie banners appearing/disappearing
- Dynamic content reordering
- Injected analytics scripts

#### Tier 3: LLM Judge (future work)
For ambiguous cases where structural hash differs but pHash is borderline, a smaller LLM could evaluate functional equivalence. Not implemented in the prototype.

**Edge case handling:**

| Scenario | How handled |
|---|---|
| Same URL, different content (SPA route) | Structural hash differs → different state |
| Same page, minor data changes | Structural hash strips dynamic text → same state |
| A/B test variant | DOM differs but pHash matches → Tier 2 catches as same state |
| Cookie banner overlay | pHash similar enough → Tier 2 catches as same state |
| Modal/drawer open vs. closed | Both tiers detect as different state (correct) |
| Pagination (page 1 vs page 2) | Same structure + same URL path → treated as same state |

### 4. Exploration Strategy

**Algorithm: Information-Gain Weighted BFS with Backtracking**

The exploration strategy combines LLM semantic understanding with a learned information-gain heuristic:

1. At each state, extract all interactive elements
2. Compute information-gain scores for each element role based on exploration history
3. Ask the LLM to rank the most promising unexplored elements, providing info-gain hints
4. Execute the top-ranked action
5. After acting, run multi-tier fingerprinting on the new page
6. If the state is new → reward the acted element's role (increases future priority)
7. If already visited → penalize the role, backtrack and try next element
8. Exploration ends when all elements are explored or max-steps reached

**Information-Gain Learning:**
- Each element role (link, button, tab, menuitem, etc.) has a reward score
- Score updates via exponential moving average: `score = score * 0.7 + reward * 0.3`
- Reward = 1.0 if the action discovered a new state, 0.0 otherwise
- Scores are passed to the LLM prompt and used as fallback ranking when LLM parsing fails

This approximates a **contextual bandit** approach — the agent learns during the session which types of interactions yield the highest exploration value, naturally deprioritizing pagination links and repeating navigation.

**Avoiding infinite loops:**
- Each (state, element) pair is tracked; once acted upon, it's marked explored
- Max steps limit provides a hard ceiling
- Information-gain scoring naturally penalizes repetitive actions

**Destructive action avoidance:**
- The LLM prompt explicitly instructs avoiding "delete", "remove", "logout", "sign out", "reset" actions
- A production system would add browser session snapshots for rollback

**Backtracking:** Direct URL navigation to parent states (more reliable than `page.goBack()` in SPAs).

### 5. Graph Model

**Node:**
```json
{
  "id": "state_001",
  "url": "https://example.com/dashboard",
  "title": "Dashboard — Overview",
  "dom_fingerprint": "a3f8c1...",
  "phash": "d4c3b2a1...",
  "interactive_elements": ["nav-link-settings", "btn-create-new"],
  "screenshot": "screenshots/state_001.png",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Edge:**
```json
{
  "from": "state_001",
  "to": "state_002",
  "action": {
    "type": "click",
    "target": "nav-link-settings",
    "selector": "a[href='/settings']",
    "description": "Clicked 'Settings' in the left nav"
  }
}
```

The graph is a **directed multigraph** — multiple edges can connect the same pair of states. Self-loops are possible (action that doesn't change state).

### 6. Observability & Telemetry

Every LLM call is recorded with:
- Provider and model used
- Input/output token counts
- Latency (ms)
- Success/failure status and error messages
- Purpose (action_selection vs. login)

Telemetry is saved alongside the graph as `*_telemetry.json`, enabling:
- **Cost tracking:** Total tokens consumed per exploration run
- **Prompt drift detection:** Monitor if success rate degrades over time
- **Debugging:** Trace exactly which LLM call failed and why

### 7. Module Structure

```
testsprite/
├── DESIGN.md              # This document
├── agent.py               # Main entry point + CLI + info-gain learning
├── core/
│   ├── __init__.py
│   ├── browser.py         # Playwright browser management
│   ├── observer.py        # Page observation + element extraction
│   ├── decider.py         # LLM integration + telemetry tracking
│   ├── executor.py        # Action execution via Playwright
│   ├── state.py           # Multi-tier state fingerprinting (hash + pHash)
│   ├── graph.py           # Graph data model + multi-tier matching
│   └── som.py             # Set-of-Mark screenshot annotation
├── screenshots/           # Captured state screenshots
└── output/                # JSON graph + telemetry output
```

### 8. Future Work (Production Hardening)

| Area | Current | Production |
|---|---|---|
| State identity | 2-tier (hash + pHash) | Add Tier 3 LLM judge for borderline cases |
| Exploration | Info-gain weighted BFS | Full RL (Q-learning / contextual bandit) with persistent policy |
| Concurrency | Single browser, serial | Containerized Playwright workers (ECS/Fargate) exploring branches in parallel |
| Queue | In-memory list | Managed message broker (Kafka/SQS) for exploration frontier |
| Page representation | Text-only by default | SoM + VLM as primary for complex SPAs |
| Destructive actions | LLM prompt avoidance | Browser session snapshots + rollback capability |
| Backtracking | URL navigation | State serialization + browser context cloning |
