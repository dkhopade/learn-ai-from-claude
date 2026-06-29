import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("rag"); // "rag" or "agent"
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendRag(question) {
    const resp = await fetch(`${API}/rag/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let firstChunk = true;
    let sources = [];
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value);

      if (firstChunk && text.includes("---")) {
        const parts = text.split("---");
        try { sources = JSON.parse(parts[0]).sources; } catch {}
        firstChunk = false;
        buffer = parts[1] || "";
      } else {
        buffer += text;
      }

      const content = buffer;
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant", content, sources, steps: [], streaming: true
        };
        return updated;
      });
    }

    setMessages(prev => {
      const updated = [...prev];
      updated[updated.length - 1].streaming = false;
      return updated;
    });
  }

  async function sendAgent(question) {
    const resp = await fetch(`${API}/agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: "ui-session" }),
    });
    const data = await resp.json();

    setMessages(prev => {
      const updated = [...prev];
      updated[updated.length - 1] = {
        role: "assistant",
        content: data.answer,
        sources: [],
        steps: data.steps || [],
        streaming: false
      };
      return updated;
    });
  }

  async function sendMessage() {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setInput("");
    setLoading(true);

    setMessages(prev => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", sources: [], steps: [], streaming: true }
    ]);

    try {
      if (mode === "rag") {
        await sendRag(question);
      } else {
        await sendAgent(question);
      }
    } catch (e) {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Error connecting to API. Is the server running?",
          sources: [], steps: [], streaming: false
        };
        return updated;
      });
    }
    setLoading(false);
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>AI Knowledge Assistant</h1>
          <p style={styles.subtitle}>
            {mode === "rag" ? "RAG mode — answers from your docs" : "Agent mode — uses tools"}
          </p>
        </div>
        <div style={styles.headerRight}>
          <div style={styles.toggle}>
            <button
              style={mode === "rag" ? styles.toggleActive : styles.toggleBtn}
              onClick={() => setMode("rag")}
            >RAG</button>
            <button
              style={mode === "agent" ? styles.toggleActive : styles.toggleBtn}
              onClick={() => setMode("agent")}
            >Agent</button>
          </div>
          <button onClick={() => setMessages([])} style={styles.clearBtn}>Clear</button>
        </div>
      </div>

      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            {mode === "rag"
              ? "Ask me about your knowledge base"
              : "Ask me anything — I can use tools (time, math, weather, knowledge base)"}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={m.role === "user" ? styles.userMsg : styles.asstMsg}>
            {m.steps?.length > 0 && (
              <div style={styles.steps}>
                <p style={styles.stepsLabel}>Tools used</p>
                {m.steps.map((s, j) => (
                  <p key={j} style={styles.stepItem}>
                    <strong>{s.tool}</strong>({JSON.stringify(s.args)}) → {s.result}
                  </p>
                ))}
              </div>
            )}
            <div style={styles.bubble(m.role)}>
              {m.content}
              {m.streaming && <span style={styles.cursor}>▊</span>}
            </div>
            {m.sources?.length > 0 && (
              <div style={styles.sources}>
                <p style={styles.sourcesLabel}>Sources</p>
                {m.sources.map((s, j) => (
                  <p key={j} style={styles.sourceItem}>{j + 1}. {s}</p>
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div style={styles.inputRow}>
        <input
          style={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          placeholder={mode === "rag" ? "Ask about your docs..." : "Ask anything, I'll use tools..."}
          disabled={loading}
        />
        <button style={styles.btn} onClick={sendMessage} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: { display: "flex", flexDirection: "column", height: "100vh", fontFamily: "system-ui, sans-serif", maxWidth: 800, margin: "0 auto", padding: "0 1rem" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1.5rem 0 1rem", borderBottom: "1px solid #eee" },
  title: { margin: 0, fontSize: 22, fontWeight: 600 },
  subtitle: { margin: "4px 0 0", fontSize: 14, color: "#888" },
  headerRight: { display: "flex", alignItems: "center", gap: 12 },
  toggle: { display: "flex", border: "1px solid #ddd", borderRadius: 8, overflow: "hidden" },
  toggleBtn: { padding: "6px 14px", fontSize: 13, border: "none", background: "#fff", cursor: "pointer", color: "#666" },
  toggleActive: { padding: "6px 14px", fontSize: 13, border: "none", background: "#0070f3", color: "#fff", cursor: "pointer" },
  clearBtn: { fontSize: 12, padding: "6px 12px", cursor: "pointer", border: "1px solid #ddd", borderRadius: 6, background: "#fff" },
  messages: { flex: 1, overflowY: "auto", padding: "1rem 0", display: "flex", flexDirection: "column", gap: 16 },
  empty: { textAlign: "center", color: "#aaa", marginTop: 80, fontSize: 15 },
  userMsg: { display: "flex", flexDirection: "column", alignItems: "flex-end" },
  asstMsg: { display: "flex", flexDirection: "column", alignItems: "flex-start" },
  bubble: (role) => ({ background: role === "user" ? "#0070f3" : "#f4f4f4", color: role === "user" ? "#fff" : "#111", padding: "10px 14px", borderRadius: 12, maxWidth: "80%", fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }),
  cursor: { display: "inline-block", animation: "blink 1s step-end infinite" },
  steps: { marginBottom: 6, maxWidth: "80%", padding: "8px 12px", background: "#f0f7ff", border: "1px solid #d0e3ff", borderRadius: 8 },
  stepsLabel: { margin: "0 0 4px", fontSize: 11, color: "#0070f3", fontWeight: 600, textTransform: "uppercase" },
  stepItem: { margin: "2px 0", fontSize: 12, color: "#444", fontFamily: "monospace" },
  sources: { marginTop: 6, maxWidth: "80%", padding: "8px 12px", background: "#fafafa", border: "1px solid #eee", borderRadius: 8 },
  sourcesLabel: { margin: "0 0 4px", fontSize: 11, color: "#888", fontWeight: 600, textTransform: "uppercase" },
  sourceItem: { margin: "2px 0", fontSize: 12, color: "#555" },
  inputRow: { display: "flex", gap: 8, padding: "1rem 0", borderTop: "1px solid #eee" },
  input: { flex: 1, padding: "10px 14px", borderRadius: 8, border: "1px solid #ddd", fontSize: 14, outline: "none" },
  btn: { padding: "10px 20px", background: "#0070f3", color: "#fff", border: "none", borderRadius: 8, fontSize: 14, cursor: "pointer" },
};
