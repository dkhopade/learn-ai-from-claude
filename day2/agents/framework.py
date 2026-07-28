"""
framework.py — a minimal multi-agent framework, built by hand.

An Agent is: a role (system prompt) + tools + a reasoning loop.
Delegation = one agent exposed as another agent's tool.

Calls go through the Track C gateway, so every agent request is
routed, cached, rate-limited, and metered automatically.

Hard-won fixes baked in (from live runs):
  Run 1: dedup guard, lenient FINAL parser, observation visibility
  Run 2: error-aware dedup, typed SUBAGENT FAILED markers, honesty rules

UI streaming support:
  Agents emit structured events (who / what / why) to a module-level
  EVENT_SINK. The backend sets the sink to stream steps to the UI via SSE.
  NOTE: a module-level sink is a deliberate simplicity tradeoff — fine for
  single-request demos; you'd redesign this (per-request context) for
  concurrent production use.
"""
import json
import os
import httpx

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")

# ── event sink: set by the caller so ALL agents' events flow one place ────
EVENT_SINK = None


def set_event_sink(fn):
    """fn(event_dict) will be called for every agent event. Pass None to clear."""
    global EVENT_SINK
    EVENT_SINK = fn


def call_llm(prompt: str, task: str = "general", max_tokens: int = 512) -> str:
    """All model access goes through the gateway — routing + caching for free."""
    resp = httpx.post(
        f"{GATEWAY_URL}/v1/chat",
        json={"prompt": prompt, "task": task, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


class Tool:
    """A named capability an agent can invoke."""
    def __init__(self, name: str, description: str, fn):
        self.name = name
        self.description = description
        self.fn = fn

    def invoke(self, argument: str) -> str:
        return str(self.fn(argument))


def _is_error(text: str) -> bool:
    """Heuristic: does an observation/result represent a failure?"""
    t = str(text).strip().upper()
    return t.startswith("ERROR") or t.startswith("SQL ERROR") or \
        t.startswith("SUBAGENT FAILED")


class Agent:
    """
    role       : who this agent is (system prompt)
    tools      : dict of Tool objects it may use
    task_type  : gateway routing hint ("general" -> base, "sql" -> sql-lora)
    max_steps  : reasoning-loop budget (prevents infinite loops)
    """
    def __init__(self, name: str, role: str, tools: dict = None,
                 task_type: str = "general", max_steps: int = 5):
        self.name = name
        self.role = role
        self.tools = tools or {}
        self.task_type = task_type
        self.max_steps = max_steps

    def _tool_manifest(self) -> str:
        if not self.tools:
            return "You have no tools."
        lines = [f"- {t.name}: {t.description}" for t in self.tools.values()]
        return "Available tools:\n" + "\n".join(lines)

    def _emit(self, kind: str, detail: str, why: str = "", verbose: bool = True):
        evt = {"agent": self.name, "kind": kind, "detail": detail, "why": why}
        if EVENT_SINK:
            EVENT_SINK(evt)
        if verbose:
            print(f"  [{self.name} {kind}] {detail[:120]}")

    def run(self, objective: str, verbose: bool = True) -> str:
        """
        The reasoning loop. Each step, the agent either:
          TOOL: <name> | <argument>     -> invoke a tool, observe result
          FINAL: <answer>               -> done
        """
        history = []
        seen_calls = {}                     # (tool, arg) -> observation (dedup)

        self._emit("start", objective, why="received objective", verbose=verbose)

        for step in range(self.max_steps):
            prompt = f"""{self.role}

{self._tool_manifest()}

Objective: {objective}

{"Previous steps:" if history else ""}
{chr(10).join(history)}

Respond with EXACTLY ONE of:
TOOL: <tool_name> | <argument>
FINAL: <your complete answer>"""

            reply = call_llm(prompt, task=self.task_type).strip()

            # Lenient parse: a FINAL anywhere wins — models sometimes emit
            # both a TOOL line and a FINAL line in one reply.
            if "FINAL:" in reply:
                answer = reply.split("FINAL:", 1)[1].strip()
                self._emit("final", answer, why="objective satisfied",
                           verbose=verbose)
                return answer

            if reply.startswith("TOOL:"):
                try:
                    body = reply[len("TOOL:"):].strip()
                    tool_name, argument = [p.strip() for p in body.split("|", 1)]
                    key = (tool_name, argument)

                    self._emit("tool_call", f"{tool_name} | {argument}",
                               why=f"step {step+1}: acting toward the objective",
                               verbose=verbose)

                    if key in seen_calls:
                        prior = seen_calls[key]
                        if _is_error(prior):
                            observation = (
                                f"REPEAT BLOCKED. That exact call already "
                                f"FAILED with: {prior}. Do NOT re-run it "
                                f"unchanged — fix the problem (for SQL: check "
                                f"your JOINs, table aliases, and column names "
                                f"against the schema) and run a CORRECTED "
                                f"version, or respond FINAL: reporting that "
                                f"you could not retrieve the data.")
                        else:
                            observation = (
                                f"REPEAT BLOCKED. You already ran this; the "
                                f"result was: {prior}. Use it and respond "
                                f"with FINAL: now.")
                    elif tool_name not in self.tools:
                        observation = f"ERROR: no tool named '{tool_name}'"
                    else:
                        observation = self.tools[tool_name].invoke(argument)
                        seen_calls[key] = observation
                except Exception as e:
                    observation = f"ERROR invoking tool: {e}"

                self._emit("observation", str(observation)[:300],
                           why="result of the tool call", verbose=verbose)
                history.append(f"Step {step+1}: called {reply}")
                history.append(f"Observation: {observation}")
                continue

            history.append(f"Step {step+1}: your reply '{reply[:60]}' was not "
                           f"valid. Use TOOL: or FINAL: exactly.")

        self._emit("error", "step budget exhausted", why="loop safety limit",
                   verbose=verbose)
        return "ERROR: step budget exhausted without a FINAL answer."


def agent_as_tool(agent: Agent, description: str) -> Tool:
    """
    THE key multi-agent move: wrap an agent so another agent can call it.

    Typed failure: if the sub-agent errors out, the caller receives an
    explicit SUBAGENT FAILED marker with instructions NOT to fabricate.
    """
    def _run(objective: str) -> str:
        result = agent.run(objective, verbose=True)
        if _is_error(result):
            return ("SUBAGENT FAILED — no data was retrieved. Do not invent "
                    "an answer from this. Either re-delegate with a "
                    "differently-worded question, or respond FINAL: stating "
                    "plainly that the data could not be obtained.")
        return result

    return Tool(name=agent.name, description=description, fn=_run)
