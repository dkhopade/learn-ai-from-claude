# Day 2 — Per-File Synopsis & Learning Guide

A walkthrough of every file built across the four Day 2 tracks, written to close the gap
between "I made this work" and "I understand why each piece exists." For each file:
**what it does**, **why it exists**, **the key concept**, and **what it taught**.

The four tracks build on each other:

- **Track A — Model serving & MLOps:** fine-tune a model and prove whether it helped
- **Track B — Observability:** see everything the system does, from GPU silicon to tokens
- **Track C — Middleware:** a gateway that routes, caches, and rate-limits model traffic
- **Track D — Multi-agent:** two agents collaborating on real tasks, visible in the UI

A recurring theme: **the hard part was rarely the happy path.** Most learning came from
failures — benchmark saturation, library API drift, container quirks, error propagation —
each diagnosed and fixed. Those are called out where they happened.

---

## Track A — Model Serving & MLOps

The goal: fine-tune Qwen2.5-7B for text-to-SQL, and — crucially — *measure whether it
actually improved*. The measurement discipline turned out to matter more than the training.

### `build_db.py`
**What it does:** Builds a small SQLite database (an "ugly" company schema — employees,
departments, projects, assignments) used as the target for evaluating text-to-SQL.

**Why it exists:** To evaluate SQL generation, you need a real database to *execute*
queries against. You can't judge SQL by how it looks — only by whether it returns the
right rows.

**Key concept — discriminating data.** The database is deliberately seeded with edge
cases: a department with no employees, an employee with no manager (NULL), two employees
tied at the top salary, a project nobody works on. Each exists so that a *wrong* query
gives a *different* answer than a right one. Without these, a buggy query and a correct
query would return identical results and the eval couldn't tell them apart.

**What it taught:** An evaluation is only as good as its data. The first version of this
DB was too "clean" — every department had employees, so a `LEFT JOIN` vs `INNER JOIN`
mistake was invisible. Rebuilding it with edge cases was the fix.

### `eval_set.py`
**What it does:** Defines the evaluation questions — natural-language questions paired
with their correct ("gold") SQL and the SQL feature each one tests (joins, subqueries,
window functions, NULL handling, etc.).

**Why it exists:** A fixed, versioned set of test questions is the "ruler" every model
version is measured against. It never changes, so scores are comparable over time.

**Key concept — adversarial evaluation.** The questions are chosen to *stress* a model,
not flatter it: correlated subqueries, the ON-vs-WHERE `LEFT JOIN` trap, ties at the
maximum, anti-joins. Each has a documented "trap" describing how a naive model fails.

**What it taught:** The first eval set was too easy — the base model scored 100%, which
means **zero headroom**: fine-tuning couldn't possibly show improvement because there was
nothing left to improve. A saturated benchmark is a broken benchmark.

### `harness.py`
**What it does:** The scoring engine. Runs a model's generated SQL against the database,
compares the *result set* to the gold query's result set, and reports two metrics:
execution accuracy (did it run?) and result-match accuracy (did it return the right rows?).

**Why it exists:** This is the objective judge. It turns "the model wrote some SQL" into
"the model was right 87.5% of the time" — a number you can trust and compare.

**Key concept — execution-based evaluation.** SQL can be written many correct ways
(`ORDER BY x LIMIT 1` vs `WHERE x = (SELECT MAX...)`), so comparing query *strings* is
wrong. Comparing *results* — the actual rows returned — is the only fair test. This is
the same methodology academic benchmarks (Spider, BIRD) use.

**What it taught:** The harness has a built-in sanity check: it scores the gold SQL
against itself and must get 100%. That check caught two bugs in the *eval's own design*
before they could corrupt real results. Lesson: an eval is software too, and needs testing.

### `predict.py`
**What it does:** Sends an eval question (plus the schema) to a model via an
OpenAI-compatible API and extracts the SQL from the response. Includes a "mock" predictor
for testing the pipeline with no model at all.

**Why it exists:** It's the bridge between the harness and the model. It also cleans up
model output — stripping markdown fences, prose, and other noise around the actual SQL.

**Key concept — separating the pipeline from the model.** The mock predictor lets you
validate the entire eval flow on CPU, for free, before spending a cent of GPU time. Only
the final scoring run needs a real GPU.

**What it taught:** Real models wrap their SQL in explanations and code fences; robust
output-parsing is a real (and underestimated) part of working with LLMs.

### `run_baseline.py`
**What it does:** The entrypoint that ties it together — builds the DB, runs the eval
through a chosen model, saves results to JSON, and logs the run to MLflow.

**Why it exists:** One command to run an evaluation and record it. `--mock` for CPU
testing, `--vllm` for the real thing, `--label` to name each run.

**Key concept — reproducible experiments.** Every run is saved with its parameters and
metrics, so "was the fine-tune better?" becomes a data question, not a guess.

**What it taught:** Measure the baseline *before* training. The baseline run revealed the
model was already strong at SQL — which reshaped the whole experiment honestly.

### `prepare_training_data.py`
**What it does:** Downloads a public text-to-SQL dataset (gretelai) and reformats each
record into the chat/instruction format the model expects for fine-tuning.

**Why it exists:** Training data must match the exact conversational shape
(system/user/assistant) the instruct model was trained on, or fine-tuning fights the model
instead of building on it.

**Key concept — instruction formatting.** An instruct model learned to follow
system/user/assistant turns. Fine-tuning data must speak that same language; the loss is
computed only on the assistant's tokens (the SQL), so the model learns the mapping.

**What it taught:** Data preparation is its own discipline — the format matters as much
as the content.

### `finetune.py`
**What it does:** The actual fine-tuning — loads Qwen2.5-7B in 4-bit, attaches small LoRA
adapters, trains only those on the SQL data, and saves the adapter (~80 MB).

**Why it exists:** To specialize the model for SQL without the cost of full fine-tuning.

**Key concept — LoRA / QLoRA.** Full fine-tuning of a 7B model needs ~60GB+ of GPU
memory — won't fit on a 24GB A10. LoRA *freezes* the model and trains tiny "adapter"
matrices instead — about 0.5% of the parameters (40M of 7.6B). QLoRA additionally loads
the frozen model in 4-bit to shrink memory ~4x. The output is a small adapter file served
*on top of* the base model, not a whole new model.

**What it taught:** Watching `trainable%: 0.527` print made the concept concrete — you're
modifying half a percent of the model and it's enough. And an 80MB adapter modifying a
15GB model is LoRA's whole value in one number. Also: library APIs drift — `trl` renamed
`max_seq_length` to `max_length`, a one-line fix found only by reading the crash.

### `tracking.py` + `test_tracking.py`
**What it does:** A thin wrapper around MLflow that logs each eval/training run's
parameters, metrics, and result files. The test script logs fake runs to verify it works.

**Why it exists:** To make experiments comparable and reproducible in a real tracking UI,
instead of scattered JSON files.

**Key concept — experiment tracking.** An experiment is only useful if it's reproducible
and comparable. MLflow turns "I think it improved" into "run #7, rank=16, 5k examples,
scored 0.875 vs baseline 0.812" — with a UI to compare side by side.

**What it taught:** Real-world setup friction is real — Python 3.14 was too new for
MLflow (had to drop to 3.12), the file-based backend was deprecated (switched to SQLite),
and `localhost` vs `127.0.0.1` mattered for the UI. None of these are in tutorials.

### The Track A result
Base Qwen scored **81.2%**; the fine-tuned adapter scored **87.5%** on the hard eval.
A small, real, *believable* improvement — and inspecting the failures showed the base
model's "mistakes" were often cosmetically-different-but-correct SQL. The honest takeaway:
a modern 7B is already strong at SQL, and the disciplined finding of "we measured, found
little headroom, and didn't ship a pointless model" is a mature MLOps outcome.

---

## Track B — Observability & Operational Excellence

The goal: see everything the system does — cluster, GPU, application, and cost — built by
hand from plain manifests so every layer is understood, not black-boxed.

### `k8s/monitoring/00-namespace.yaml`
**What it does:** Creates a dedicated `monitoring` namespace for the observability stack.

**Why it exists:** Isolating monitoring from application workloads is the realistic
pattern — it separates permissions, resource limits, and makes the whole stack easy to
manage or tear down as a unit.

**Key concept — namespaces as boundaries.** Kubernetes namespaces are logical partitions;
keeping monitoring separate from `default` (where the app runs) is basic operational hygiene.

### `k8s/monitoring/01-prometheus-rbac.yaml`
**What it does:** Grants Prometheus cluster-wide *read* permission to discover what to
monitor — nodes, pods, endpoints, services.

**Why it exists:** Prometheus uses "service discovery" — it asks the Kubernetes API what
exists and builds its scrape list dynamically. That API access requires explicit permission.

**Key concept — RBAC (ServiceAccount + ClusterRole + ClusterRoleBinding).** Three objects:
an *identity* (ServiceAccount), a set of *permissions* (ClusterRole), and the *binding*
that ties them together. Cluster-scoped because node discovery spans the whole cluster.

**What it taught:** Without the right ServiceAccount wired to the pod, service discovery
silently finds nothing — the permission model is load-bearing, not boilerplate.

### `k8s/monitoring/02-prometheus-config.yaml`
**What it does:** The scrape configuration — tells Prometheus *what* to monitor and how
often: itself, the nodes (via kubelet), per-container metrics (via cAdvisor), and any pod
that opts in with an annotation.

**Why it exists:** This is Prometheus's brain. It defines both static targets and dynamic
service-discovery jobs.

**Key concept — the pull model + opt-in annotations.** Prometheus *scrapes* (pulls)
metrics on an interval, rather than receiving them. The clever part: any pod annotated
`prometheus.io/scrape: "true"` gets discovered and scraped automatically — so later, DCGM,
vLLM, and the gateway all joined monitoring *without touching this config again*.

**What it taught:** The `relabel_configs` that rewrite discovery metadata into scrape
addresses (routing kubelet scrapes through the API-server proxy) are the standard trick
for monitoring private-subnet nodes — fiddly, but the pattern is reusable.

### `k8s/monitoring/03-prometheus-deploy.yaml`
**What it does:** Runs the Prometheus server — reads the config, discovers targets, scrapes
them, and stores the metrics as time-series.

**Why it exists:** It's the metrics database and the engine that populates it.

**Key concept — connecting pod to permissions.** The `serviceAccountName: prometheus`
line is what links the running pod to the RBAC permissions. Metrics use `emptyDir`
storage (ephemeral) — fine for learning; a PVC would make history survive restarts.

**What it taught:** The CRI-O "short image name" lesson from Day 1 applied again — images
must be fully qualified (`docker.io/prom/prometheus:...`). Pattern recognition paying off.

### `k8s/monitoring/04-grafana.yaml`
**What it does:** Runs Grafana (the dashboards), pre-wired to Prometheus as its data source
via a config file so no manual setup is needed.

**Why it exists:** Grafana turns raw time-series numbers into readable, live dashboards.

**Key concept — Grafana queries, doesn't store.** Grafana holds no data; it runs PromQL
queries against Prometheus and renders the results. The data source is auto-provisioned
via a mounted ConfigMap — Grafana reads it on startup and wires the connection itself.

**What it taught:** Separate pods talk over the cluster network, so Grafana reaches
Prometheus by its *service DNS name* (`prometheus.monitoring.svc:9090`), not `localhost`.
That distinction was the first bug to fix.

### `k8s/monitoring/05-dcgm-exporter.yaml`
**What it does:** Runs NVIDIA's DCGM exporter on the GPU node, exposing GPU telemetry —
utilization, memory, temperature, power — in Prometheus format.

**Why it exists:** This is the "nvidia-smi as a dashboard" payoff. It turns manual terminal
squinting into live, historical, graphed GPU metrics.

**Key concept — DaemonSet + annotations.** A DaemonSet runs one pod per matching node
(here, GPU nodes only, via nodeSelector + toleration). The `prometheus.io/scrape`
annotation makes Prometheus auto-discover it — the opt-in pattern from the config paying
off with zero config changes.

**What it taught:** GPU metrics only flow when the GPU is up; the exporter's pod sits
`Pending` when the GPU node is gone, and springs to life when it returns — expected, not
broken.

### The token-economics dashboard (built in Grafana UI, exported to JSON)
**What it does:** Ties GPU utilization (from DCGM) and token throughput (from vLLM) to a
fixed GPU hourly cost, surfacing cost-per-million-tokens and idle-spend.

**Why it exists:** On accelerators, cost *is* a performance metric. A GPU costs the same
per hour whether busy or idle — the economic question is how many useful tokens you get
per dollar.

**Key concept — FinOps for inference.** High utilization + high throughput = low
cost-per-token = efficient. Low utilization = paying full price for little output. The
dashboard makes that tradeoff *visible* — cost-per-token drops as throughput rises.

**What it taught:** Watching GPU power/temperature climb as a model loaded, then
utilization spike during inference, made the physical reality of ML infrastructure
tangible — and connected directly to cost.

---

## Track C — Middleware & API Layer

The goal: a gateway between clients and the model(s) that adds routing, caching, rate
limiting, and metrics — the same middleware patterns from classic web engineering, applied
to LLM serving.

### `gateway/app.py`
**What it does:** A FastAPI service that sits in front of vLLM. It receives requests,
*routes* them to the right model (base vs. fine-tuned sql-lora), checks a *cache* before
calling the GPU, enforces a *rate limit*, and emits Prometheus *metrics* on everything.

**Why it exists:** In production, clients shouldn't talk to the model directly. A gateway
is the control point for cost, safety, and observability — one front door with policy.

**Key concepts (four, layered):**
- **Routing:** a request's `task` field decides the model. `task=sql` → the fine-tuned
  adapter; everything else → base model. Model selection becomes an *architectural policy*,
  not a per-request detail.
- **Caching:** before calling the GPU, hash the (model, prompt, params) and check an
  in-memory cache. A hit returns instantly with *zero GPU time* — every hit is money saved.
  Only caches deterministic (temperature=0) requests, since caching random output is wrong.
- **Rate limiting:** a token-bucket limiter allows short bursts but caps sustained rate,
  rejecting excess with a cheap HTTP 429 instead of overwhelming the GPU.
- **Metrics:** every request, cache event, and rejection is counted and labeled, feeding
  the Track B Grafana stack.

**What it taught:** Each capability was validated on CPU with a mock upstream before any
GPU spend — routing decisions, cache hit/miss/bypass logic, and rate-limit rejections all
proven for free. The cache-hit *savings* were then demonstrated live: the same request
twice — first a GPU spike, then an instant cached response with the GPU dashboard flat.

### `gateway/Dockerfile` + `k8s/gateway.yaml`
**What it does:** Containerizes the gateway and deploys it to the cluster on CPU nodes,
annotated so Prometheus scrapes its metrics automatically.

**Why it exists:** To run the gateway as a real cluster service that the backend (and
agents) call over in-cluster DNS.

**Key concept — the whole stack composes.** The gateway is scraped by Track B's Prometheus
(via the same annotation pattern), serves models from Track A, and gets consumed by Track
D's agents. Four tracks, one system.

**What it taught:** The `--host 0.0.0.0` containerization gotcha (a server must listen on
all interfaces inside a container, not just localhost) and the SHA-pinned-image deployment
trap (a deployment pinned to an old commit SHA will faithfully re-pull the *old* image
forever, regardless of restarts — the fix is `kubectl set image` with the new SHA).

---
## Track D — Multi-Agent Systems

The goal: two specialized agents collaborating on a real task — an Analyst that orchestrates
and a SQL Specialist that queries a database — with their reasoning shown live in the UI.

### `agents/framework.py`
**What it does:** A minimal, hand-built multi-agent framework. Defines an `Agent` (a role +
tools + a reasoning loop), a `Tool` (a named capability), and — the key move —
`agent_as_tool`, which wraps one agent so another can call it.

**Why it exists:** To understand the actual machinery of agents from first principles,
rather than importing a framework that hides it.

**Key concepts:**
- **The reasoning loop:** each step, the agent's model emits either `TOOL: <name> | <arg>`
  (act and observe) or `FINAL: <answer>` (done). Reason → act → observe → repeat.
- **Delegation = composition:** `agent_as_tool` makes the Specialist a *callable capability*
  of the Analyst. Multi-agent coordination isn't magic — it's one agent's tool being
  another agent. That's the whole idea in five lines.
- **Event streaming:** agents emit structured events (who/what/why) to a sink, so the
  reasoning can be streamed to a UI (used in Track D's UI integration).
- **All calls route through the Track C gateway** — so agent traffic is cached, rate-limited,
  routed per-agent to the right model, and metered in Grafana.

**What it taught:** This file is where the deepest lessons landed, through *failure*:
- **Loop detection:** models re-issue identical tool calls; a dedup guard blocks repeats.
- **Protocol robustness:** models don't reliably follow output formats — sometimes emitting
  both a TOOL and a FINAL line. The parser had to be lenient (FINAL wins).
- **Error propagation — the big one:** in one run, a sub-agent's SQL error propagated up
  and the orchestrator *hallucinated a confident, wrong answer from it* — inventing
  departments that don't exist. This is *the* failure mode of multi-agent systems: errors
  don't just fail, they get laundered into plausible fiction as they pass between agents.
  The fixes — typed failure markers (`SUBAGENT FAILED`), error-aware dedup ("that failed,
  fix it" not "use this result"), and an explicit honesty rule ("never invent data") —
  turned "error → fiction" into "error → honest failure or self-repair." Every serious
  agent framework's complexity exists because of exactly this.

### `agents/team.py`
**What it does:** Defines the two concrete agents and the real SQL-execution tool.
- **Specialist:** uses the fine-tuned `sql-lora` model (via gateway routing); its tool
  executes SQL against the Track A database.
- **Analyst:** uses the base model; its tool is the Specialist. It delegates data questions,
  interprets results for a business audience, and refuses to fabricate.

**Why it exists:** This is where the whole project converges — the fine-tuned model from
Track A gets a production role, called through the Track C gateway, observable via Track B,
coordinated by the framework.

**Key concept — right model for each role.** The Specialist routes to the fine-tuned SQL
adapter; the Analyst routes to the base model for general reasoning. Two models, two roles,
selected by *architecture*.

**What it taught:** The before/after was stark on the same anti-join question: *before* the
fixes, confident fabrication ("no unassigned employees" — false); *after*, correct SQL
(the Specialist even repaired its own missing JOIN) and a truthful answer. Same models —
the difference was entirely the *engineering around* them.

### `agent_routes.py` (backend) + `AgentTeamPanel.js` + `App.js` (frontend)
**What it does:** Wires the agent team into the existing app. A backend SSE endpoint runs
the team and streams each reasoning step to the browser; a React panel renders the steps as
a live, color-coded timeline (who acted, what they did, why), then the final answer.

**Why it exists:** To turn the agents from a terminal script into a self-explaining product
feature — the user watches the agents think.

**Key concept — SSE streaming + the queue bridge.** The agents run synchronously and *push*
events; SSE wants to *pull* and yield them. A queue bridges the two: the agent thread pushes
events in, the SSE generator pulls them out and streams each to the browser as it happens.

**What it taught:** Real integration gotchas — Python package imports differ between running
a script directly (`from framework import`) and importing it as a package
(`from .framework import`), requiring a try/except import and an `__init__.py`. And a build
context change (repo root vs. subfolder) was needed so the backend image could include the
agents code.

---

## The Meta-Lessons of Day 2

Beyond any single file, the recurring themes worth carrying forward:

1. **Measure before you optimize.** The eval harness, built first, revealed the model was
   already good — saving a pointless fine-tune. You can't improve what you can't measure,
   and you can't trust a measurement you haven't validated.

2. **Inspect failures, never just the aggregate.** An 81.2% score looked like weakness;
   inspecting the failures showed correct-but-different SQL. The number was true; the naive
   interpretation would have been false.

3. **The ecosystem drifts under you.** `torch_dtype`→`dtype`, `trl`'s `max_seq_length`
   rename, MLflow's deprecated file store, Python 3.14 being too new — each a one-line fix
   once diagnosed. Working at the frontier means libraries move faster than tutorials.

4. **Containers are minimal; assume nothing.** `git` missing, `python` vs `python3`,
   fully-qualified image names, `--host 0.0.0.0` — the environment is spare by design.

5. **Preserve your evidence.** A too-aggressive Job TTL deleted training logs before they
   could be read. Observability applies to your own build process, not just the app.

6. **Errors compound in distributed/multi-agent systems.** A single-agent error is visible;
   a multi-agent error gets laundered into confident fiction. Typed failures and honesty
   rules aren't optional polish — they're the core engineering.

7. **Cost is a first-class metric.** GPU discipline — build on CPU, GPU only for the
   irreducible work, tear down immediately — ran through every track. On accelerators,
   cost and performance are two views of the same thing.

**The whole system, at the end:** one vLLM serving base + fine-tuned LoRA + tool-calling;
a gateway routing/caching/metering all traffic; full observability from GPU to token to
dollar; and a multi-agent team — all running together, three UI modes (RAG, Agent, Team)
live in production. Built by hand, debugged from real failures, understood layer by layer.
