# Memory Strategies: Approach and Trade-offs

## The problem

Out of the box, conversational AI agents are stateless. Every session starts from a blank slate — the user's name, their project context, the decisions they made three conversations ago, all of it gone. For a one-off tool this is fine. For anything meant to feel like a genuine assistant, it's a real limitation.

The goal here was to implement three distinct cross-conversation memory strategies, run them through the same scripted scenario, and compare them honestly across the dimensions that actually matter in production: recall accuracy, token cost, and latency overhead.

---

## How the evaluation works

The harness runs three sessions against each strategy:

- **Session 1** plants specific, non-inferable facts — the user's name, their company, the internal name of the service they're building, their SLA, their rollback window.
- **Session 2** floods memory with a second batch of unrelated information (a fraud detection service, a currency normalisation service) to dilute what was stored in Session 1.
- **Session 3** asks direct recall questions about Session 1 facts: "What's my name?", "What's the service called?", "What's our SLA?".

These questions are deliberately simple but impossible to answer without memory. There's no domain knowledge or reasonable guessing that gets you to "Sarah Chen" or "LedgerSync" or "200ms". Either the memory system surfaced the right context, or it didn't.

An LLM judge scores each answer 0–5 based on how many expected facts were present. Any hedge like "I don't have access to our prior conversations" is an instant zero, regardless of how politely it's worded.

---

## The three strategies

### Full History

Every user and assistant message is stored verbatim in SQLite and replayed in full on every request. It's the simplest possible implementation — no summarisation, no embeddings, no intelligence. Just an append-only log.

The good news is that it's completely lossless. If the user mentioned something three months and fifty sessions ago, it's still there, word for word. The bad news is that "still there" means it gets injected into every prompt, in full, forever. A user who's had 100 sessions will have an enormous context injection on their 101st, and eventually you'll hit the model's context window limit. Cost also grows linearly with usage, which doesn't work at scale.

**When it makes sense:** Internal tools with a small number of power users, short-lived projects, or anywhere you need guaranteed perfect recall and aren't worried about growth.

### Summary Memory

After each turn, an LLM call merges the new exchange into a running summary of everything known about the user. The summary gets updated in place — it doesn't grow with sessions, it stays compact.

This is clever in theory. The prompt injection stays small regardless of how many sessions have happened, and the summary often captures the important things well. The problem is that LLM summarisation is inherently lossy. Specific numbers and exact names are exactly the kind of detail that summaries smooth over. And the update happens synchronously — every store call blocks for several seconds while the summarisation LLM runs. In our results that was ~7 seconds per turn. That's fine for a background process, but it's a lot of overhead to absorb.

**When it makes sense:** Long-lived users where you care more about the big picture than exact recall of specific details. Works well when the user's context is mostly narrative ("they're working on a payments migration") rather than precise facts ("their SLA is exactly 200ms").

### Semantic Memory

Each exchange is converted to an embedding vector and stored. At retrieval time, the current query is embedded and compared against everything stored using cosine similarity. Only the top-K most relevant chunks come back, which means the prompt injection stays bounded regardless of how much history accumulates.

This is the most scalable approach. The 1,000th session costs roughly the same as the 10th, because you're only ever injecting the most relevant chunks. Retrieval and storage both take a few hundred milliseconds — enough to be noticeable but nothing that blocks the user experience.

The one real risk is that retrieval is query-dependent. If the user asks something that doesn't semantically resemble how the relevant memory was originally phrased, the right chunk might not rank highly enough to get retrieved. History and summary don't have this problem — they inject everything. Semantic memory is making a bet that relevance and recency are good proxies for "what the user needs right now", which is usually true but not always.

---

## Results

```
Metric                            none     history     summary    semantic
--------------------------------------------------------------------------
Mean judge score (0-5)             0.0         5.0         5.0         5.0
Mean prompt tokens                 193        4136        1417        2591
Token delta vs none           baseline       +3944       +1225       +2398
Mean retrieve ms                     0           1           1         351
Mean store ms                        0           2        7296         368
Total overhead (s)                0.00        0.01       21.89        2.16
```

All three memory strategies scored 5.0 — perfect recall across all questions even after Session 2 injected noise. The stateless baseline scored 0.0, as expected: it simply can't answer questions about things it was never told.

The differentiation shows up entirely in cost and overhead. History consumed 4,136 prompt tokens per turn — 21x the baseline and 3x more than semantic. Summary kept tokens down to 1,417 but paid for it in store latency: 7.3 seconds per turn for the summarisation call. Semantic sits in the middle on tokens (2,591) with consistent sub-400ms overhead on both sides.

---

## My recommendation: Semantic Memory

For a production system, semantic memory is the right default.

History fails at scale. Token cost growing without bound is a design problem you'll eventually have to solve, and context window limits mean it simply breaks for long-lived users. It's a great starting point but not a shipping architecture.

Summary is compelling on token efficiency but the synchronous LLM-on-every-store pattern is hard to justify. Seven seconds of background overhead per turn adds up, and you can't fully trust the summariser to preserve every specific fact the user cared about.

Semantic memory sidesteps both problems. Storage is cheap, retrieval is bounded, and the latency overhead is low enough that it doesn't meaningfully affect the user experience. The query-dependence limitation is real, but it's addressable — you can hedge against it by also storing a lightweight structured summary for identity and preferences, and reserving semantic retrieval for the richer conversational context.

If I were productionising this, I'd run summary and semantic together: a small structured record for facts that should always be available (name, role, key constraints), and semantic retrieval for everything else. You get the bounded cost of semantic with the reliable recall of a curated summary for the facts that matter most.
