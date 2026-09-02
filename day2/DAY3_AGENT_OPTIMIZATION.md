# Day 3 — Agent Optimization

Day 2 built the system. Day 3 asks a different question: **what does one user question
actually cost, and where does that cost go?**

The unit matters. A chat request is one model call. An *agent* request is a reasoning
loop — delegate, query, observe, restate — and every step is a full prompt sent to a GPU.
Measuring cost per *request* on an agentic system tells you almost nothing. The unit that
matters is **cost per completed task**.

Everything below was measured on the Day 2 platform: Qwen2.5-7B + sql-lora on a single
A10, served by vLLM, fronted by the Track C gateway, on OKE.

**Scope, stated up front:** single node, single model, a synthetic 4-table SQLite schema,
5–15 questions per experiment, no repeats per condition. These are directional findings
on a small system, not production benchmarks. Where a result is a negative result or an
un-validated hypothesis, it says so.

---

## The instrumentation

### `agentmetrics.py`
**What it does:** Records every LLM call inside one agent run — which agent made it,
prompt size, response size, latency, and whether the gateway served it from cache. Writes
a per-run aggregate (`results/agent_baseline.csv`) and a per-call detail file
(`results/agent_calls.csv`).

**Why it exists:** Day 2's observability measured the *system* — GPU utilization, tokens,
cost per million. None of it could answer "how many model calls did that one question
take?" That question lives above the serving layer and below the application, and nothing
was watching it.

**Key concept — instrument at the choke point.** Every model call in the Day 2 framework
goes through one function, `call_llm()`. Instrumenting there captures the whole loop
without touching agent logic, tool definitions, or the reasoning parser.

**What it taught:** The gateway was already returning a `cached` flag in every response
and the agent code was discarding it. The single most useful metric in this whole exercise
was data the system had been emitting all along, unread.

**Design note — subprocesses, not threads.** The collectors are module-level globals, the
same deliberate simplicity tradeoff `framework.py` notes for `EVENT_SINK`. The concurrency
harness runs each session as a subprocess so each gets its own module state. Threads would
race on `start_run()` and corrupt each other's rows.

---

## Finding 1 — Orchestration cost 50% of calls and bought output consistency, not correctness

**The experiment:** run five questions through the full Analyst → Specialist team, then
run the same five through the Specialist alone (`agents/solo.py`), cold cache both times.

**Result (medians across 5 questions):**

| | Team | Specialist only | Delta |
|---|---|---|---|
| LLM calls | 4 | 2 | **−50%** |
| Prompt chars sent | 6,095 | 3,380 | **−44%** |
| LLM latency | 4,802 ms | 2,825 ms | **−41%** |

The SQL was correct in both conditions on all five questions. The fine-tuned adapter wrote
a working query first try every time — no repair loops fired, which is Track A's 87.5%
eval score showing up as reliability rather than as a number.

**What the orchestration layer actually bought:** consistency of the output contract. The
Analyst reliably produced business-language answers. The Specialist alone produced them
about half the time and returned raw tuples — `(7,)`, `(30,)` — the rest. Same agent, same
prompt, run to run.

**Key concept — the tradeoff is a routing decision, not a verdict.** If a human reads the
answer, the Analyst earns its cost. If a service consumes the result, it is 50% waste.
"Multi-agent is overhead" is the wrong conclusion; "orchestration should be conditional on
the consumer" is the right one.

**Caveat:** five questions, one run each, on a synthetic schema. The formatting
inconsistency is an observation, not a measured rate.

---

## Finding 2 — The capacity ceiling was admission control, not the GPU

**The experiment:** `concurrency.py` runs N agent sessions in parallel at N = 1, 2, 4, 8.
Fifteen distinct questions across the sweep, no repeats. The gateway's rate limiter was
deliberately left enabled.

**Result:**

| Concurrency | Sessions completed | Gateway 429s | Throughput (sessions/s) | Prefix cache hit |
|---|---|---|---|---|
| 1 | 1 / 1 | 0 | 0.209 | 97.7% |
| 2 | 2 / 2 | 0 | 0.285 | 97.3% |
| 4 | 2 / 4 | 4 | 0.363 | 96.1% |
| 8 | 1 / 8 | 11 | 0.210 | 97.1% |

**An A10 serving a 7B model fell over at two concurrent users — and the GPU was never the
constraint.**

**Key concept — rate limits sized in the wrong unit.** The gateway allows 2 requests/second
with a burst of 5. That is a sensible chat limit, where one user is one request. But one
*agent* session is four gateway calls. Two concurrent sessions is eight calls in a few
seconds; eight sessions is thirty-two against a bucket sized for five. The limiter should
have been counting sessions, not requests. This generalizes: any admission control tuned
for conversational traffic will collapse under agentic traffic by roughly the loop depth.

**The negative result, and it matters.** The prefix cache hit rate was expected to *decline*
under concurrency as interleaved sessions evicted each other's KV blocks. It did not — it
held near 97% at every level. The reason is that the rate limiter throttled load before
vLLM's cache ever came under pressure. Admission control protected the GPU so effectively
that the contention this experiment was designed to observe never occurred. The explanation
is more useful than the measurement would have been.

**Caveat:** "failed" sessions here are processes that died on a 429, because these agents
have no retry logic. A production client would back off and retry, so real user-visible
behaviour would be added latency, not failure.

---

## Finding 3 — Two caches, same traffic, opposite outcomes

Measured during the runs above, not as a separate experiment:

- **Gateway cache hit rate under agent load: 0%** — zero hits across every call of every
  baseline run.
- **vLLM automatic prefix cache: ~97% per run** (cumulative 5,616 hits / 7,731 queried
  tokens = 72.6% across a mixed session).

**Key concept — cache granularity is the whole story.** The gateway cache hashes
`(model, prompt, params)` and asks *"have I seen this exact prompt?"* Inside an agent loop
the answer is always no, because history accumulates on every step. vLLM asks *"have I seen
this token prefix?"* and the answer is usually yes, because role, tool manifest, and schema
are identical every step.

Whole-prompt caching is the wrong unit for agentic workloads. Prefix-level reuse is what
actually matters — which is why KV-cache-aware routing is the interesting problem in
multi-turn agent serving, and why cross-instance KV reuse is where the serving frameworks
are investing.

**A property the loop got right, worth naming:** the agent appends history to the *end* of
the prompt. Anything varying at the *start* — a timestamp, a session ID, a shuffled tool
list — would collapse the prefix hit rate to near zero. Prompt construction order is a
performance decision.

---

## Finding 4 — Orchestration was 38% of GPU time, and none of it needed a 7B model

**The experiment:** per-call breakdown across 5 fresh questions, 22 total LLM calls,
classifying each Analyst call as either the initial *delegate* decision or the final
*restate* step.

| Step | Calls | Share of GPU time | Median latency | Median output |
|---|---|---|---|---|
| Analyst — delegate | 5 | 11.0% | 717 ms | 68 chars |
| Analyst — restate | 5 | 27.1% | 1,617 ms | 191 chars |
| Specialist — SQL | 12 | 61.8% | 1,575 ms | 109 chars |

**Orchestration: 10 of 22 calls, 38.2% of total GPU time.**

**The delegate step is routing, not reasoning.** 717 ms to emit 68 characters selecting
which tool to call. A 1B model would do this. An embedding classifier would do it without
a model at all.

**The restate step is 27% of GPU time spent on formatting.** The Specialist has already
produced the correct answer; the Analyst re-reads it and rewrites it in business English.
It is the slowest orchestration step precisely because it *generates* — and decode is
memory-bound, one token at a time, which is exactly the profile where a smaller model wins
most.

**Key concept — right-sizing per step.** The entire loop ran on the largest model served,
because it was the *only* model served. Model selection per step is the lever the gateway's
routing layer already exists to pull; it just wasn't being pulled.

**Caveat — this sizes the opportunity, it does not validate the fix.** No smaller model was
served and no before/after was measured. The honest claim is "38% of GPU time goes to work
that does not obviously need a 7B model," not "routing to a small model would save 38%."

**Also worth noting:** orchestration cost is *fixed* at two calls regardless of question
difficulty, while Specialist cost varies (4 or 5 calls depending on whether it self-repaired).
So orchestration overhead is proportionally worst on the easiest questions.

---

## Methodology — what went wrong, and what it taught

More time went into making the measurements trustworthy than into taking them. That ratio
is the lesson.

**1. Warm cache contaminated three separate experiments.** The first solo comparison showed
a 10× speedup that was entirely the gateway serving answers warmed by the baseline runs. The
first concurrency sweep reused questions across levels, so higher concurrency looked
*faster*. Each time the fix was the same: flush the cache, use questions never run before.

**A system with caching cannot be benchmarked by repetition.** Obvious in hindsight; caught
three times only because the numbers looked too good.

**2. The harness had a concurrency bug the system didn't.** Every child process called
`build()`, which runs `CREATE TABLE`. Concurrent processes raced and all but one died with
`table depts already exists`. Fixed by building once in the parent and setting
`SKIP_DB_BUILD=1` for children. The bug was in the measurement tool, not the thing measured —
worth being precise about that distinction.

**3. A benchmark that reports zeros as a result is worse than one that crashes.** One sweep
returned a clean, well-formatted table showing 0 sessions succeeded at every level, because
both port-forwards had died silently. The success check — exit code 0 — was too weak; a
subprocess can exit cleanly having done nothing. It now requires the run summary to have
actually printed. **Instrumentation needs its own failure modes considered, or it will
confidently report nothing as something.**

**4. Failure-path instrumentation earned its keep immediately.** Recording metrics in a
`finally` block meant a run that died on a 429 still produced a row. Failed runs are often
the expensive ones.

---

## What this changes about the system

Ordered by measured impact, with honesty about what is proven:

1. **Rate-limit by session, not request** — *measured, not fixed.* The current limiter caps
   the platform at two concurrent users regardless of GPU headroom.
2. **Route the delegate step to a small model or a classifier** — *sized, not validated.*
   11% of GPU time producing 68 characters of routing.
3. **Make orchestration conditional on the consumer** — *measured.* 50% of calls buy output
   formatting that a machine consumer does not need.
4. **Move the gateway cache to prefix or semantic granularity** — *measured.* Exact-match
   hashing has a structurally zero hit rate inside an agent loop.

None of these were shipped. This was a measurement exercise, and the output is a ranked,
sized list of what to do next — which is the point.

---

## The meta-lesson

Day 2's closing theme was that cost is a first-class metric. Day 3 sharpens it:

**On agentic systems, the unit of cost is the completed task, and most of the interesting
inefficiency lives between the application and the serving layer — in loop depth,
orchestration necessity, cache granularity, and admission control sized for the wrong kind
of traffic.** None of that is visible from GPU utilization graphs, and none of it shows up
if you measure per request.
