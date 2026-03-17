"""
Scale Harness — Needle-in-Haystack Memory Test

Tests each memory strategy's ability to recall a single planted fact (the "needle")
after being flooded with N noise facts from a pre-generated bank.

The needle is a UUID-based code injected into the middle of the noise batch.
Scoring is a binary string match — no LLM judge needed. Either the strategy
surfaced the needle or it didn't.

Prerequisites:
    Generate the fact bank first (one-time):
        uv run python -m evals.generate_fact_bank

Then run the scale harness:
    uv run python -u -m evals.harness_scale
    uv run python -u -m evals.harness_scale | tee harness_scale_output.txt

Noise levels and strategies can be configured via NOISE_LEVELS and STRATEGIES below.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.memory import get_memory_strategy

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGIES = ["none", "history", "summary", "semantic", "agentic"]
NOISE_LEVELS = [10, 25, 50, 100]
FACTS_PER_TURN = 5        # How many noise facts to bundle per conversation turn
CHAT_MODEL = "anthropic:claude-haiku-4-5-20251001"
USER_ID = "scale_user"
DB_PATH = "harness_scale_memory.db"
FACT_BANK_PATH = Path(__file__).parent / "fact_bank.json"

# ---------------------------------------------------------------------------
# Needle construction
# ---------------------------------------------------------------------------

# A single needle code is generated per run and reused across all strategies
# and noise levels, so results are directly comparable.
_NEEDLE_CODE = uuid.uuid4().hex[:8].upper()
_NEEDLE_FACT = (
    f"The internal hotfix reference code for the payment rollback procedure is {_NEEDLE_CODE}."
)
_NEEDLE_QUESTION = "What is the internal hotfix reference code for the payment rollback procedure?"

# ---------------------------------------------------------------------------
# Shared chat helper (same as main harness — single LLM round-trip)
# ---------------------------------------------------------------------------

_BASE_SYSTEM = (
    "You are a helpful AI assistant. "
    "When memory from prior conversations is provided in this prompt, treat it as your "
    "own verified recall — use it directly and confidently to answer questions. "
    "Do not disclaim about lacking persistent memory; the memory section is authoritative."
)


def _chat(messages: list, memory_context: str | None = None) -> tuple[str, int, int]:
    """Single LLM round-trip. Returns (response_text, prompt_tokens, completion_tokens)."""
    system_content = _BASE_SYSTEM
    if memory_context:
        system_content = (
            f"The following is verified memory from prior conversations with this user. "
            f"Treat it as your own recall and use it confidently:\n\n"
            f"{memory_context}\n\n"
            f"{_BASE_SYSTEM}"
        )
    model = init_chat_model(CHAT_MODEL)
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


# ---------------------------------------------------------------------------
# Fact bank loading
# ---------------------------------------------------------------------------

def load_fact_bank(n_needed: int) -> list[str]:
    if not FACT_BANK_PATH.exists():
        raise FileNotFoundError(
            f"Fact bank not found at {FACT_BANK_PATH}.\n"
            "Generate it first with: uv run python -m evals.generate_fact_bank"
        )
    facts = json.loads(FACT_BANK_PATH.read_text())
    if len(facts) < n_needed:
        raise ValueError(
            f"Fact bank has {len(facts)} facts but {n_needed} are needed. "
            "Re-run generate_fact_bank.py to add more."
        )
    return facts[:n_needed]


# ---------------------------------------------------------------------------
# Core test logic
# ---------------------------------------------------------------------------

def _build_turns(noise_facts: list[str]) -> list[str]:
    """
    Assemble noise facts into batched turns, with the needle injected at the midpoint.

    Facts are grouped FACTS_PER_TURN per turn to simulate realistic conversation
    density. The needle is inserted as a standalone turn in the middle so it isn't
    trivially at the most-recent or least-recent position.
    """
    mid = len(noise_facts) // 2
    before = noise_facts[:mid]
    after = noise_facts[mid:]

    turns = []
    for i in range(0, len(before), FACTS_PER_TURN):
        batch = before[i : i + FACTS_PER_TURN]
        turns.append(" ".join(batch))

    turns.append(_NEEDLE_FACT)  # needle planted at midpoint

    for i in range(0, len(after), FACTS_PER_TURN):
        batch = after[i : i + FACTS_PER_TURN]
        turns.append(" ".join(batch))

    return turns


def run_scale_test(
    strategy,
    noise_facts: list[str],
) -> "ScaleResult":
    """
    Store all noise turns + needle, then ask the needle question.
    Returns recall (bool), prompt tokens, retrieve_ms, and store_ms.
    """
    turns = _build_turns(noise_facts)
    total_store_ms = 0.0

    # --- Accumulation: store all turns (no LLM retrieval needed here) ---
    for turn_text in turns:
        t0 = time.time()
        strategy.store(USER_ID, [
            {"role": "user", "content": turn_text},
            {"role": "assistant", "content": "Noted."},
        ])
        total_store_ms += (time.time() - t0) * 1000

    # AgenticMemory updates core memory asynchronously — wait before recall.
    if hasattr(strategy, "wait_for_core_update"):
        strategy.wait_for_core_update()

    # --- Recall: retrieve memory and ask the needle question ---
    t0 = time.time()
    memory_context = strategy.retrieve(USER_ID, _NEEDLE_QUESTION)
    retrieve_ms = (time.time() - t0) * 1000

    response, prompt_tokens, _ = _chat(
        [{"role": "user", "content": _NEEDLE_QUESTION}],
        memory_context or None,
    )

    recalled = _NEEDLE_CODE.lower() in response.lower()

    return ScaleResult(
        recalled=recalled,
        response_snippet=response[:200],
        prompt_tokens=prompt_tokens,
        retrieve_ms=retrieve_ms,
        total_store_ms=total_store_ms,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScaleResult:
    recalled: bool
    response_snippet: str
    prompt_tokens: int
    retrieve_ms: float
    total_store_ms: float


@dataclass
class StrategyScaleResults:
    name: str
    results: dict[int, ScaleResult] = field(default_factory=dict)  # noise_level -> result

    def recall_str(self, noise_level: int) -> str:
        if noise_level not in self.results:
            return "—"
        return "✓" if self.results[noise_level].recalled else "✗"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DIVIDER = "=" * 76


def print_results(all_results: list[StrategyScaleResults]) -> None:
    print(f"\n{DIVIDER}")
    print("  SCALE TEST — NEEDLE-IN-HAYSTACK RESULTS")
    print(DIVIDER)
    print(f"  Needle code:  {_NEEDLE_CODE}")
    print(f"  Question:     {_NEEDLE_QUESTION}")
    print(f"  Facts/turn:   {FACTS_PER_TURN}  |  Needle position: midpoint of batch")
    print(DIVIDER)

    col_w = 12
    name_w = 14
    noise_cols = "".join(f"{f'{n} facts':>{col_w}}" for n in NOISE_LEVELS)

    def row(label, values):
        return f"{label:<{name_w}}" + "".join(f"{v:>{col_w}}" for v in values)

    print(f"\n{'Recall (✓/✗)':<{name_w}}{noise_cols}")
    print("-" * (name_w + col_w * len(NOISE_LEVELS)))
    for r in all_results:
        print(row(r.name, [r.recall_str(n) for n in NOISE_LEVELS]))

    print(f"\n{'Prompt tokens':<{name_w}}{noise_cols}")
    print("-" * (name_w + col_w * len(NOISE_LEVELS)))
    for r in all_results:
        values = []
        for n in NOISE_LEVELS:
            res = r.results.get(n)
            values.append(str(res.prompt_tokens) if res else "—")
        print(row(r.name, values))

    print(f"\n{'Retrieve ms':<{name_w}}{noise_cols}")
    print("-" * (name_w + col_w * len(NOISE_LEVELS)))
    for r in all_results:
        values = []
        for n in NOISE_LEVELS:
            res = r.results.get(n)
            values.append(f"{res.retrieve_ms:.0f}" if res else "—")
        print(row(r.name, values))

    print(f"\n{'Store total s':<{name_w}}{noise_cols}")
    print("-" * (name_w + col_w * len(NOISE_LEVELS)))
    for r in all_results:
        values = []
        for n in NOISE_LEVELS:
            res = r.results.get(n)
            values.append(f"{res.total_store_ms / 1000:.1f}" if res else "—")
        print(row(r.name, values))

    print()

    # Per-strategy detail
    print(f"\n{DIVIDER}")
    print("  RESPONSE DETAIL")
    print(DIVIDER)
    for r in all_results:
        print(f"\n[{r.name}]")
        for n in NOISE_LEVELS:
            res = r.results.get(n)
            if not res:
                continue
            status = "✓ RECALLED" if res.recalled else "✗ MISSED"
            print(f"  {n:>3} facts — {status}")
            print(f"  Response: {res.response_snippet}{'...' if len(res.response_snippet) == 200 else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    max_noise = max(NOISE_LEVELS)
    print(f"Loading fact bank (need {max_noise} facts)...")
    fact_bank = load_fact_bank(max_noise)
    print(f"Loaded {len(fact_bank)} facts from {FACT_BANK_PATH.name}")

    print(f"\n{DIVIDER}")
    print("  Scale Harness — Needle-in-Haystack Memory Test")
    print(f"  Model: {CHAT_MODEL}")
    print(f"  Needle: {_NEEDLE_CODE}")
    print(f"  Strategies: {', '.join(STRATEGIES)}")
    print(f"  Noise levels: {NOISE_LEVELS}")
    print(DIVIDER)

    db = Path(DB_PATH)

    all_results: list[StrategyScaleResults] = []

    for strategy_name in STRATEGIES:
        strategy_results = StrategyScaleResults(name=strategy_name)

        for noise_level in NOISE_LEVELS:
            # Fresh DB for every (strategy, noise_level) combination
            if db.exists():
                db.unlink()

            strategy = get_memory_strategy(strategy_name, db_path=DB_PATH)
            strategy.clear(USER_ID)

            noise_facts = fact_bank[:noise_level]

            print(f"[{strategy_name}] {noise_level} facts...", end=" ", flush=True)

            if strategy_name == "none":
                # Stateless baseline — skip accumulation, recall will always fail
                t0 = time.time()
                memory_context = strategy.retrieve(USER_ID, _NEEDLE_QUESTION)
                retrieve_ms = (time.time() - t0) * 1000
                response, prompt_tokens, _ = _chat(
                    [{"role": "user", "content": _NEEDLE_QUESTION}],
                    memory_context or None,
                )
                result = ScaleResult(
                    recalled=_NEEDLE_CODE.lower() in response.lower(),
                    response_snippet=response[:200],
                    prompt_tokens=prompt_tokens,
                    retrieve_ms=retrieve_ms,
                    total_store_ms=0.0,
                )
            else:
                result = run_scale_test(strategy, noise_facts)

            status = "✓" if result.recalled else "✗"
            print(f"{status}  ({result.prompt_tokens} tokens, retrieve {result.retrieve_ms:.0f}ms)")
            strategy_results.results[noise_level] = result

        all_results.append(strategy_results)

    # Clean up DB after run
    if db.exists():
        db.unlink()

    print_results(all_results)


if __name__ == "__main__":
    main()
