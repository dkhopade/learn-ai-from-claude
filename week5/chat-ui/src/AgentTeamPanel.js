import React, { useState, useRef } from "react";

// Icons per event kind — the visual language of the timeline
const KIND_ICONS = {
  start: "🧠",
  tool_call: "⚙️",
  observation: "📊",
  final: "✅",
  answer: "🎯",
  error: "⚠️",
};

const KIND_LABELS = {
  start: "received objective",
  tool_call: "invoking tool",
  observation: "observed result",
  final: "concluded",
  answer: "FINAL ANSWER",
  error: "error",
};

export default function AgentTeamPanel({ apiBase }) {
  const [question, setQuestion] = useState("");
  const [events, setEvents] = useState([]);
  const [running, setRunning] = useState(false);
  const [answer, setAnswer] = useState(null);
  const bottomRef = useRef(null);

  const ask = async () => {
    if (!question.trim() || running) return;
    setEvents([]);
    setAnswer(null);
    setRunning(true);

    try {
      const resp = await fetch(`${apiBase}/agent-team/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by double newlines
        const frames = buffer.split("\n\n");
        buffer = frames.pop(); // keep the incomplete tail

        for (const frame of frames) {
          if (!frame.startsWith("data: ")) continue;
          const evt = JSON.parse(frame.slice(6));
          if (evt.kind === "done") continue;
          if (evt.kind === "answer") {
            setAnswer(evt.detail);
          }
          setEvents((prev) => [...prev, evt]);
          bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        }
      }
    } catch (e) {
      setEvents((prev) => [
        ...prev,
        { agent: "system", kind: "error", detail: String(e), why: "" },
      ]);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: 16 }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          style={{ flex: 1, padding: "10px 12px", fontSize: 15,
                   borderRadius: 8, border: "1px solid #ccc" }}
          value={question}
          placeholder="Ask a business question about the company data…"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          disabled={running}
        />
        <button
          onClick={ask}
          disabled={running || !question.trim()}
          style={{ padding: "10px 18px", borderRadius: 8, border: "none",
                   background: running ? "#999" : "#c74634", color: "#fff",
                   fontWeight: 600, cursor: "pointer" }}
        >
          {running ? "Thinking…" : "Ask the team"}
        </button>
      </div>

      {/* The reasoning timeline */}
      <div>
        {events.map((evt, i) => (
          <div key={i}
            style={{
              display: "flex", gap: 10, padding: "8px 12px", marginBottom: 6,
              borderRadius: 8, fontSize: 13.5, lineHeight: 1.5,
              background: evt.kind === "answer" ? "#eef7ee"
                        : evt.kind === "error" ? "#fdecea" : "#f6f6f4",
              borderLeft: `3px solid ${
                evt.agent === "analyst" ? "#3a6ea5"
                : evt.agent === "sql_specialist" ? "#7d57c1" : "#c74634"}`,
            }}>
            <span>{KIND_ICONS[evt.kind] || "•"}</span>
            <div>
              <strong>{evt.agent}</strong>{" "}
              <span style={{ color: "#777" }}>
                {KIND_LABELS[evt.kind] || evt.kind}
              </span>
              <div style={{ fontFamily: evt.kind === "tool_call"
                            ? "monospace" : "inherit",
                            whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {evt.detail}
              </div>
              {evt.why && (
                <div style={{ fontSize: 11.5, color: "#999", marginTop: 2 }}>
                  why: {evt.why}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {answer && (
        <div style={{ marginTop: 16, padding: 16, borderRadius: 10,
                      background: "#eef7ee", border: "1px solid #cde5cd",
                      fontSize: 15 }}>
          <strong>🎯 Answer:</strong> {answer}
        </div>
      )}
    </div>
  );
}
