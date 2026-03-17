"""
Fact Bank Generator

Generates 500 unique engineering facts about a fictional payments company and
saves them to evals/fact_bank.json. Used by harness_scale.py as noise data.

Run once before using the scale harness:
    uv run python -m evals.generate_fact_bank

Each batch of 50 facts is generated with a running list of already-used values
(numbers, proper nouns) passed to the next prompt, preventing any value from
appearing more than once across all 500 facts.
"""

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

OUTPUT_PATH = Path(__file__).parent / "fact_bank.json"
MODEL = "anthropic:claude-haiku-4-5-20251001"
BATCH_SIZE = 50
TARGET = 500

# Words that must not appear in any fact — reserved for the needle in harness_scale.py
_BLOCKED_WORDS = {"incident", "rollback", "hotfix", "reference code", "ledgersync", "tracking code"}

_BATCH_PROMPT = """\
Generate {n} distinct engineering facts about a fictional payments software company.

Rules:
- Every fact is exactly one sentence.
- No two facts may share any number, service name, team name, technology, or metric \
value — every value must be unique across all facts.
- Use invented proper nouns for all services, teams, databases, and tools. \
Do NOT use real product names (no Kafka, PostgreSQL, Redis, AWS, GCP, etc.).
- Do NOT mention any of the following topics: incident codes, rollback procedures, \
hotfix references, tracking codes, or error codes.
- Do NOT use any of the values listed below (already used in prior batches).

Already-used values (do not repeat any of these):
{used_values}

Return a JSON array of exactly {n} strings. No preamble, no markdown fences."""


def _extract_values(facts: list[str]) -> set[str]:
    """Extract numbers and invented proper nouns (CamelCase/compound names) from facts.

    We deliberately skip plain sentence-start capitals ("The", "Our", "Team")
    because those are common English words, not unique values. Only numbers and
    names that look invented (CamelCase like PayVault, or multi-word like
    IridiumDB) count as values that must not repeat.
    """
    values: set[str] = set()
    for fact in facts:
        # Numbers with optional units: 847, 3.2, 99.94%, 200ms
        values.update(re.findall(r"\b\d+(?:\.\d+)?(?:ms|s|%)?\b", fact))
        # CamelCase words — likely invented service/team names (e.g. PayVault, LedgerSync)
        values.update(re.findall(r"\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b", fact))
        # ALL-CAPS abbreviations (e.g. SLA, API, GCP)
        values.update(re.findall(r"\b[A-Z]{3,}\b", fact))
    return values


def _is_safe(fact: str) -> bool:
    low = fact.lower()
    return not any(w in low for w in _BLOCKED_WORDS)


def _generate_batch(model, n: int, used_values: set[str]) -> list[str]:
    used_str = ", ".join(sorted(used_values)) if used_values else "(none yet)"
    prompt = _BATCH_PROMPT.format(n=n, used_values=used_str)
    result = model.invoke(
        [{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str) and f.strip() and _is_safe(f)]
    except json.JSONDecodeError:
        pass
    return []


def _deduplicate(facts: list[str]) -> list[str]:
    """Remove exact duplicates and any fact whose numbers/invented names overlap with another."""
    seen_strings: set[str] = set()
    seen_values: set[str] = set()
    result = []
    for fact in facts:
        if fact in seen_strings:
            continue
        values = _extract_values([fact])
        if values & seen_values:
            continue
        seen_strings.add(fact)
        seen_values.update(values)
        result.append(fact)
    return result


def main():
    model = init_chat_model(MODEL)
    all_facts: list[str] = []
    used_values: set[str] = set()
    batches = TARGET // BATCH_SIZE

    print(f"Generating {TARGET} facts in {batches} batches of {BATCH_SIZE}...")

    for i in range(batches):
        print(f"  Batch {i + 1}/{batches} (used values pool: {len(used_values)})...", end=" ", flush=True)
        batch = _generate_batch(model, BATCH_SIZE, used_values)
        batch = [f for f in batch if f not in set(all_facts)]
        all_facts.extend(batch)
        used_values.update(_extract_values(batch))
        print(f"got {len(batch)} facts (total: {len(all_facts)})")

    print(f"\nDeduplicating {len(all_facts)} raw facts...")
    all_facts = _deduplicate(all_facts)
    print(f"After dedup: {len(all_facts)} facts")

    if len(all_facts) < TARGET:
        print(f"WARNING: Only generated {len(all_facts)} facts (target was {TARGET}).")
        print("Re-run the script to top up, or lower TARGET in harness_scale.py.")

    OUTPUT_PATH.write_text(json.dumps(all_facts, indent=2))
    print(f"\nSaved {len(all_facts)} facts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
