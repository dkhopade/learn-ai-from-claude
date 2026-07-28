"""
agent_routes.py — SSE endpoint that runs the Day 2 agent team and streams
each reasoning step to the UI as it happens.

Bridge pattern: agents push events -> queue -> SSE generator pulls -> browser.
"""
import json
import queue
import threading

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# the day2 agents package (copied into the image at build time)
from agents.framework import set_event_sink
from agents.team import analyst, build

router = APIRouter()


class AgentQuestion(BaseModel):
    question: str


@router.post("/agent-team/stream")
def agent_team_stream(req: AgentQuestion):
    build()  # ensure the eval DB exists in the container

    q: "queue.Queue[dict]" = queue.Queue()

    def sink(evt: dict):
        q.put(evt)

    def run_team():
        set_event_sink(sink)
        try:
            answer = analyst.run(req.question, verbose=False)
            q.put({"agent": "system", "kind": "answer",
                   "detail": answer, "why": "final result"})
        except Exception as e:
            q.put({"agent": "system", "kind": "error",
                   "detail": str(e), "why": "unhandled failure"})
        finally:
            set_event_sink(None)
            q.put({"kind": "done"})

    threading.Thread(target=run_team, daemon=True).start()

    def event_stream():
        while True:
            evt = q.get()
            if evt.get("kind") == "done":
                yield "data: " + json.dumps({"kind": "done"}) + "\n\n"
                break
            yield "data: " + json.dumps(evt) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
