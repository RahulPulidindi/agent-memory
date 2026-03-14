"""
Memory Strategy Evaluation Harness

Runs a scripted three-session enterprise conversation against each memory strategy
and evaluates responses across three dimensions:

  1. LLM-as-Judge    (0-5 recall quality score)
  2. Token Efficiency (exact prompt tokens from API response metadata)
  3. Latency Overhead (wall-clock ms for retrieve() and store())

Usage:
    uv run python -m evals.harness
    uv run python -m evals.harness | tee harness_output.txt
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.memory import get_memory_strategy

load_dotenv()

# ---------------------------------------------------------------------------
# Scripted conversations
# ---------------------------------------------------------------------------

# Session 1: plants specific, non-inferable facts about the user and their project.
SESSION_1 = [
    (
        "I'm Sarah Chen, principal engineer on the Core Payments Platform team at Meridian "
        "Financial. We're migrating our monolithic Ruby on Rails payment processor to a "
        "Python-based microservices architecture. Happy to finally have a proper AI assistant "
        "to think through this with."
    ),
    (
        "Quick context on our stack: primary database is PostgreSQL 15, we use Redis Cluster "
        "for distributed caching, and all services run on GKE — Google Kubernetes Engine. "
        "Our end-to-end SLA for the reconciliation pipeline is 200ms."
    ),
    (
        "The new reconciliation service is internally called LedgerSync. It needs to emit "
        "structured audit events to our Kafka cluster on every reconciliation operation. "
        "What's the cleanest way to structure that event emission?"
    ),
    (
        "Good point on schema evolution. We've decided to go with event sourcing for the "
        "audit trail, using Apache Avro for schema enforcement across all audit events. "
        "Our hard cutover deadline is end of Q3, and we've committed to a 48-hour rollback "
        "window if LedgerSync has production issues."
    ),
]

# Session 2: floods memory with different information to dilute Session 1 details.
# Lossy strategies (summary compression, mismatched semantic chunks) may drop early facts.
SESSION_2_ACCUMULATION = [
    (
        "New topic: we're also building a separate fraud detection service called "
        "RiskGuard. It runs ML inference using a PyTorch model, scores transactions in "
        "real time, and writes flagged events to a separate Elasticsearch index."
    ),
    (
        "RiskGuard has a latency budget of 50ms per transaction and is deployed as a "
        "sidecar on every payment pod. The model is retrained weekly using Kubeflow "
        "Pipelines. The team owns a separate on-call rotation — they're SRE-embedded."
    ),
    (
        "We also have a currency normalisation service called FXBridge. It fetches live "
        "rates from three providers — Bloomberg, Reuters, and an internal treasury feed — "
        "caches them for 90 seconds in Redis, and falls back to the prior rate on errors. "
        "FXBridge exposes a synchronous gRPC API to LedgerSync."
    ),
    (
        "FXBridge SLO is 99.95% availability with a p99 latency of 30ms. The treasury "
        "feed is the authoritative source; the others are fallbacks in priority order. "
        "Rate discrepancies above 0.5% trigger an automated alert to the FX desk."
    ),
]

# Session 3: asks about Session 1 facts after Session 2 noise has been injected.
# Questions have unique correct answers — impossible to answer without prior memory.
SESSION_3 = [
    (
        "Going back to basics — what's my name, role, and company?",
        {
            "description": "Recall of user identity after memory dilution by Session 2",
            "expected_facts": ["Sarah Chen", "principal engineer", "Meridian Financial"],
        },
    ),
    (
        "Remind me: what's the internal name of the reconciliation service and what's "
        "its audit event serialization format?",
        {
            "description": "Recall of LedgerSync name and Avro format after Session 2 noise",
            "expected_facts": ["LedgerSync", "Apache Avro"],
        },
    ),
    (
        "What was our agreed reconciliation SLA and rollback window?",
        {
            "description": "Recall of SLA and rollback window after Session 2 dilution",
            "expected_facts": ["200ms", "48-hour"],
        },
    ),
]

PLANTED_FACTS = """
Session 1 (early, may be diluted by Session 2 in lossy strategies):
- User: Sarah Chen, principal engineer, Core Payments Platform team, Meridian Financial
- Database: PostgreSQL 15 | Cache: Redis Cluster | Deployment: GKE
- Reconciliation SLA: 200ms end-to-end
- Service name: LedgerSync, publishes audit events to Kafka
- Audit pattern: event sourcing with Apache Avro schema enforcement
- Cutover: end of Q3 | Rollback window: 48 hours

Session 2 (recent, higher retrieval weight in some strategies):
- Fraud service: RiskGuard, PyTorch ML, 50ms budget, Elasticsearch, Kubeflow Pipelines
- Currency service: FXBridge, gRPC to LedgerSync, Bloomberg/Reuters/treasury feeds
- FXBridge: 90-second Redis cache, 99.95% SLO, p99 30ms, 0.5% discrepancy alert
""".strip()

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an impartial evaluator checking whether an AI assistant correctly recalled \
specific facts from a prior conversation session.

For each question you are given a short list of EXPECTED FACTS. \
Your only job is to check how many appear correctly in the assistant's response.

Apply these rules in strict order — stop at the first that matches:

RULE 1 — INSTANT ZERO: If the response contains any hedge or refusal \
("I don't recall", "I don't have access to", "I don't have that information", \
"no prior memory", or similar), SCORE: 0. No exceptions.

RULE 2 — INSTANT FIVE: If every expected fact is present and accurate \
(semantic equivalents count — "Principal Engineer" equals "principal engineer", \
"48 hours" equals "48-hour"), SCORE: 5. Extra context the assistant volunteers \
beyond the expected facts does not reduce the score.

RULE 3 — PARTIAL CREDIT: If some expected facts are missing, use the rubric below.

Return your response as exactly two lines:
SCORE: <integer 0-5>
REASON: <one sentence listing which expected facts were present and which were missing>"""

JUDGE_RUBRIC = """\
Partial-credit rubric (apply only when RULE 1 and RULE 2 do not trigger):
1 - Fewer than 40% of expected facts present and accurate.
2 - 40–59% of expected facts present and accurate.
3 - 60–79% of expected facts present and accurate.
4 - 80–99% of expected facts present and accurate (all but one minor detail missing).
5 - 100% present — reachable only via RULE 2 above."""


def judge_response(
    question: str, response: str, expected_facts: list[str], model_str: str
) -> tuple[int, str]:
    """Score a recall response using an LLM judge."""
    model = init_chat_model(model_str)
    facts_line = ", ".join(f'"{f}"' for f in expected_facts)
    prompt = (
        f"{JUDGE_RUBRIC}\n\n"
        f"All facts established in prior sessions:\n{PLANTED_FACTS}\n\n"
        f"Expected facts for THIS question:\n{facts_line}\n\n"
        f"Question:\n{question}\n\n"
        f"Assistant response:\n{response}"
    )
    result = model.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    score, reason = 0, "Could not parse judge response."
    for line in result.content.strip().splitlines():
        if line.startswith("SCORE:"):
            try:
                score = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    question: str
    description: str
    response: str
    judge_score: int
    judge_reason: str
    prompt_tokens: int
    completion_tokens: int
    retrieve_ms: float
    store_ms: float


@dataclass
class StrategyResult:
    name: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def mean_judge_score(self) -> float:
        return sum(t.judge_score for t in self.turns) / len(self.turns) if self.turns else 0.0

    @property
    def mean_prompt_tokens(self) -> float:
        return sum(t.prompt_tokens for t in self.turns) / len(self.turns) if self.turns else 0.0

    @property
    def mean_retrieve_ms(self) -> float:
        return sum(t.retrieve_ms for t in self.turns) / len(self.turns) if self.turns else 0.0

    @property
    def mean_store_ms(self) -> float:
        return sum(t.store_ms for t in self.turns) / len(self.turns) if self.turns else 0.0

    @property
    def total_overhead_s(self) -> float:
        return sum(t.retrieve_ms + t.store_ms for t in self.turns) / 1000


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------

STRATEGIES = ["none", "history", "summary", "semantic", "agentic"]
CHAT_MODEL = "anthropic:claude-haiku-4-5-20251001"
JUDGE_MODEL = "anthropic:claude-haiku-4-5-20251001"
USER_ID = "harness_user"
DB_PATH = "harness_memory.db"

# Used by the harness directly — bypasses create_deep_agent's tool-use loop,
# which adds 3-8 extra LLM calls per turn and would make the harness 5-10x slower.
_BASE_SYSTEM = (
    "You are a helpful AI assistant engaging in a technical conversation. "
    "Respond helpfully and concisely. "
    "When memory from prior conversations is provided in this prompt, treat it as your "
    "own verified recall — use it directly and confidently to answer questions. "
    "Do not disclaim about lacking persistent memory; the memory section is authoritative."
)


def _chat(
    model_str: str,
    messages: list,
    memory_context: str | None = None,
) -> tuple[str, int, int]:
    """Single LLM round-trip. Returns (response_text, prompt_tokens, completion_tokens)."""
    system_content = _BASE_SYSTEM
    if memory_context:
        system_content = (
            f"The following is verified memory from prior conversations with this user. "
            f"Treat it as your own recall and use it confidently and directly in your answers:\n\n"
            f"{memory_context}\n\n"
            f"{_BASE_SYSTEM}"
        )

    model = init_chat_model(model_str)
    result = model.invoke([{"role": "system", "content": system_content}] + messages)

    prompt_tokens = completion_tokens = 0
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
    elif usage is not None:
        prompt_tokens = getattr(usage, "input_tokens", 0)
        completion_tokens = getattr(usage, "output_tokens", 0)

    return result.content, prompt_tokens, completion_tokens


def _run_accumulation_session(turns: list[str], strategy, model_str: str) -> None:
    """Run a scripted session and persist every turn. No retrieval — this IS the memory source."""
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        ai_content, _, _ = _chat(model_str, messages)
        messages.append({"role": "assistant", "content": ai_content})
        strategy.store(USER_ID, [
            {"role": "user", "content": turn},
            {"role": "assistant", "content": ai_content},
        ])
    # AgenticMemory updates core memory asynchronously — wait for it to finish
    # before starting the next session so the core block is ready for retrieval.
    if hasattr(strategy, "wait_for_core_update"):
        strategy.wait_for_core_update()


def run_recall_session(
    turns: list[tuple[str, dict]],
    strategy,
    model_str: str,
    judge_model: str,
) -> list[TurnResult]:
    """Run a recall session and measure all three metrics per turn."""
    messages = []
    turn_results = []

    for question, meta in turns:
        t0 = time.time()
        memory_context = strategy.retrieve(USER_ID, question)
        retrieve_ms = (time.time() - t0) * 1000

        messages.append({"role": "user", "content": question})
        response_text, prompt_tokens, completion_tokens = _chat(
            model_str, messages, memory_context or None
        )
        messages.append({"role": "assistant", "content": response_text})

        t0 = time.time()
        strategy.store(USER_ID, [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_text},
        ])
        store_ms = (time.time() - t0) * 1000

        judge_score, judge_reason = judge_response(
            question, response_text, meta["expected_facts"], judge_model
        )
        turn_results.append(
            TurnResult(
                question=question,
                description=meta["description"],
                response=response_text,
                judge_score=judge_score,
                judge_reason=judge_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                retrieve_ms=retrieve_ms,
                store_ms=store_ms,
            )
        )

    return turn_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DIVIDER = "=" * 72


def print_strategy_report(result: StrategyResult) -> None:
    print(f"\n{DIVIDER}")
    print(f"  Strategy: {result.name.upper()}")
    print(DIVIDER)
    for i, turn in enumerate(result.turns, 1):
        print(f"\nSession 3, Turn {i}: \"{turn.description}\"")
        print(f"  Q: {turn.question[:120]}{'...' if len(turn.question) > 120 else ''}")
        print(f"  A: {turn.response[:300]}{'...' if len(turn.response) > 300 else ''}")
        print(f"  Judge:   {turn.judge_score}/5  --  {turn.judge_reason}")
        print(f"  Tokens:  {turn.prompt_tokens} prompt / {turn.completion_tokens} completion")
        print(f"  Latency: retrieve {turn.retrieve_ms:.0f}ms | store {turn.store_ms:.0f}ms")


def print_comparison_table(results: list[StrategyResult]) -> None:
    none_tokens = next((r.mean_prompt_tokens for r in results if r.name == "none"), 0)
    col_w, name_w = 12, 26

    print(f"\n{DIVIDER}")
    print("  SUMMARY COMPARISON")
    print(DIVIDER)
    print(f"{'Metric':<{name_w}}" + "".join(f"{r.name:>{col_w}}" for r in results))
    print("-" * (name_w + col_w * len(results)))

    def row(label, values):
        return f"{label:<{name_w}}" + "".join(f"{v:>{col_w}}" for v in values)

    print(row("Mean judge score (0-5)", [f"{r.mean_judge_score:.1f}" for r in results]))
    print(row("Mean prompt tokens", [f"{r.mean_prompt_tokens:.0f}" for r in results]))
    print(row("Token delta vs none", [
        "baseline" if r.name == "none" else f"+{r.mean_prompt_tokens - none_tokens:.0f}"
        for r in results
    ]))
    print(row("Mean retrieve ms", [f"{r.mean_retrieve_ms:.0f}" for r in results]))
    print(row("Mean store ms", [f"{r.mean_store_ms:.0f}" for r in results]))
    print(row("Total overhead (s)", [f"{r.total_overhead_s:.2f}" for r in results]))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(DIVIDER)
    print("  Memory Strategy Evaluation Harness")
    print(f"  Model: {CHAT_MODEL}")
    print(f"  Judge: {JUDGE_MODEL}")
    print(f"  Strategies: {', '.join(STRATEGIES)}")
    print(DIVIDER)

    db = Path(DB_PATH)
    if db.exists():
        db.unlink()

    all_results: list[StrategyResult] = []

    for strategy_name in STRATEGIES:
        strategy = get_memory_strategy(strategy_name, db_path=DB_PATH)
        strategy.clear(USER_ID)

        if strategy_name == "none":
            print(f"\n[{strategy_name}] Skipping accumulation sessions (stateless baseline).")
        else:
            print(f"\n[{strategy_name}] Session 1: planting initial facts...")
            _run_accumulation_session(SESSION_1, strategy, CHAT_MODEL)
            print(f"[{strategy_name}] Session 2: injecting dilution noise...")
            _run_accumulation_session(SESSION_2_ACCUMULATION, strategy, CHAT_MODEL)

        print(f"[{strategy_name}] Session 3: testing recall of Session 1 facts...")
        turns = run_recall_session(SESSION_3, strategy, CHAT_MODEL, JUDGE_MODEL)

        result = StrategyResult(name=strategy_name, turns=turns)
        all_results.append(result)
        print(f"[{strategy_name}] Done. Mean judge score: {result.mean_judge_score:.1f}/5")

    for result in all_results:
        print_strategy_report(result)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()