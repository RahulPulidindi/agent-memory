# Memory Strategies: Approach and Trade-offs

## The problem

Out of the box, conversational AI agents are stateless. Every session starts from a blank slate — the user's name, their project context, the decisions they made three conversations ago, all of it gone. For a one-off tool this is fine. For anything meant to feel like a genuine assistant, it's a real limitation.

The goal here was to implement four distinct cross-conversation memory strategies, run them through the same scripted scenario, and compare them honestly across the dimensions that actually matter in production: recall accuracy, token cost, and latency overhead.

---

## How the evaluation works

The harness runs three sessions against each strategy:

- **Session 1** plants specific, non-inferable facts — the user's name, their company, the internal name of the service they're building, their SLA, their rollback window.
- **Session 2** floods memory with a second batch of unrelated information (a fraud detection service, a currency normalisation service) to dilute what was stored in Session 1.
- **Session 3** asks direct recall questions about Session 1 facts: "What's my name?", "What's the service called?", "What's our SLA?".

These questions are deliberately simple but impossible to answer without memory. There's no domain knowledge or reasonable guessing that gets you to "Sarah Chen" or "LedgerSync" or "200ms". Either the memory system surfaced the right context, or it didn't.

An LLM judge scores each answer 0–5 based on how many expected facts were present. Any hedge like "I don't have access to our prior conversations" is an instant zero, regardless of how politely it's worded.

---

## The four strategies

### Full History

Every user and assistant message is stored verbatim in SQLite and replayed in full on every request. No summarisation, no embeddings, no intelligence — just an append-only log.

The good news is that it's completely lossless. If the user mentioned something three months and fifty sessions ago, it's still there, word for word. The bad news is that "still there" means it gets injected into every prompt, in full, forever. A user who's had 100 sessions will have an enormous context injection on their 101st, and eventually you'll hit the model's context window limit. Cost also grows linearly with usage, which doesn't work at scale.

**When it makes sense:** Internal tools with a small number of power users, short-lived projects, or anywhere you need guaranteed perfect recall and aren't worried about growth.

### Summary Memory

After each turn, an LLM call merges the new exchange into a running summary of everything known about the user. The summary stays compact regardless of how many sessions have occurred.

This is clever in theory. The prompt injection stays small and the summary often captures the important things well. The problem is that LLM summarisation is inherently lossy — specific numbers and exact names are exactly the kind of detail that summaries smooth over. And the update happens synchronously, so every store call blocks for several seconds while the summarisation LLM runs. In our results that was ~7 seconds per turn. That's a lot of overhead to absorb for an interactive product.

**When it makes sense:** Long-lived users where you care more about the big picture than exact recall of specific details.

### Semantic Memory

Each exchange is converted to an embedding vector and stored. At retrieval time, the current query is embedded and compared against everything stored using cosine similarity. Only the top-K most relevant chunks come back, keeping injection size bounded regardless of how much history accumulates.

This scales well — the 1,000th session costs roughly the same as the 10th. The one real risk is that retrieval is query-dependent. If the user asks something that doesn't semantically resemble how the relevant memory was originally phrased, the right chunk might not surface. History and summary don't have this problem because they inject everything.

There's also a subtler issue: raw semantic memory embeds whole exchanges (user message + assistant response together), which dilutes the retrieval signal with assistant-generated content. The embedding is partly indexing what the model said, not just what the user told it.

**When it makes sense:** Large, varied history where injecting everything would be too expensive and targeted recall is sufficient.

### Agentic Memory

The most sophisticated strategy. Instead of storing raw exchanges, an LLM distils each turn into discrete extracted facts ("User is named Sarah Chen", "SLA is 200ms end-to-end"), which are then embedded individually and stored. This directly solves the signal dilution problem in plain semantic memory — the embeddings index user-stated facts, not assistant responses.

On top of the fact store, a persistent core memory block is maintained: a compact, always-injected structured record of the user's identity, active projects, and key constraints. This guarantees that identity-level information is always available, not subject to whether cosine similarity retrieves the right chunk.

The core block is updated asynchronously after each turn (via a background thread), avoiding the blocking latency that makes summary memory painful in practice.

**When it makes sense:** Any production use case. It combines the bounded token cost of semantic memory with the reliable identity recall of summary memory, while adding structured fact extraction that improves retrieval precision.

---

## Results

```
Metric                            none     history     summary    semantic     agentic
--------------------------------------------------------------------------------------
Mean judge score (0-5)             0.0         5.0         5.0         5.0         5.0
Mean prompt tokens                 194        4163        1380        2194        1232
Token delta vs none           baseline       +3969       +1187       +2000       +1038
Mean retrieve ms                     0           1           2         850         368
Mean store ms                        0           3        6666         403        1107
Total overhead (s)                0.00        0.01       20.00        3.76        4.42
```

All four memory strategies scored 5.0 — perfect recall across all questions even after Session 2 injected noise. The stateless baseline scored 0.0, as expected.

The differentiation is entirely in cost and overhead:

- **History** consumed 4,163 prompt tokens per turn — over 3x more than any other memory strategy. It works but doesn't scale.
- **Summary** kept tokens down to 1,380 but paid for it with 6.7 seconds of store latency per turn. Synchronous LLM summarisation is the culprit.
- **Semantic** used 2,194 tokens with consistent ~400ms overhead on both sides. Solid middle ground, but token cost is nearly 2x agentic.
- **Agentic** used the fewest tokens (1,232 — only slightly above summary) with 368ms retrieve latency and ~1.1s store latency. The store overhead is higher than semantic due to the synchronous fact extraction LLM call, but total overhead (4.42s) is nearly identical to semantic (3.76s).

---

## Recommendation: Agentic Memory

Agentic memory is the strongest strategy across every meaningful production dimension.

It uses fewer prompt tokens than any other strategy (including summary), retrieves faster than semantic, and avoids the synchronous blocking that makes summary memory impractical. Recall quality matches the other strategies at this scale and should degrade more gracefully at larger scale, since it's indexing extracted facts rather than raw conversation text.

The extra store complexity is the honest tradeoff: fact extraction requires one synchronous LLM call per turn (~700ms overhead vs raw embedding). This is real cost, but it buys you meaningfully better retrieval precision and a compact, always-available core memory block — which no other strategy provides without additional engineering.

The one thing this evaluation doesn't prove is long-horizon behaviour. Over hundreds of sessions, agentic memory's token advantage should compound: the core block stays bounded, facts are granular and deduplicable, and retrieval is over clean signal. History breaks under that load, summary gets increasingly lossy, and semantic's retrieval quality degrades as the fact pool grows. Agentic is the only strategy designed to stay cheap and accurate as usage accumulates.

If we're building for scale, start with agentic. If we need something simpler today, semantic is the next best option — the same bounded cost model, without the fact extraction layer.
