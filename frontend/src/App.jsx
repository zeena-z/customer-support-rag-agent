import { useState } from "react";
import {
  LayoutDashboard,
  MessageSquare,
  Send,
  Ticket,
  Clock3,
  CheckCircle2,
  AlertTriangle,
  Bot,
  User,
  Sparkles,
  RefreshCw,
} from "lucide-react";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [activePage, setActivePage] = useState("dashboard");

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <h1>Support AI</h1>
            <span>Intelligence Platform</span>
          </div>
        </div>

        <nav className="navigation">
          <button
            className={`nav-item ${
              activePage === "dashboard" ? "active" : ""
            }`}
            onClick={() => setActivePage("dashboard")}
          >
            <LayoutDashboard size={20} />
            <span>Dashboard</span>
          </button>

          <button
            className={`nav-item ${activePage === "chat" ? "active" : ""}`}
            onClick={() => setActivePage("chat")}
          >
            <MessageSquare size={20} />
            <span>AI Chat</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="status-dot"></div>
          <div>
            <strong>System Online</strong>
            <span>AI services available</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        {activePage === "dashboard" ? <Dashboard /> : <Chat />}
      </main>
    </div>
  );
}

/* =========================================================
   DASHBOARD
========================================================= */

function Dashboard() {
  const [ticketText, setTicketText] = useState("");
  const [triageResult, setTriageResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleTriage() {
    if (!ticketText.trim()) return;

    setLoading(true);
    setError("");
    setTriageResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/triage`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          subject: "Support Ticket",
          body: ticketText,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Server returned ${response.status}: ${errorText}`
        );
      }

      const data = await response.json();
      setTriageResult(data);
    } catch (err) {
      console.error("Triage error:", err);
      setError(
        "Could not connect to the backend. Make sure your FastAPI server is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">SUPPORT INTELLIGENCE</p>
          <h2>Dashboard</h2>
          <p className="page-description">
            Monitor support operations and intelligently triage incoming
            tickets.
          </p>
        </div>

        <div className="header-badge">
          <div className="status-dot"></div>
          API Connected
        </div>
      </header>

      {/* STAT CARDS */}

      <section className="stats-grid">
        <StatCard
          icon={<Ticket size={22} />}
          title="Total Tickets"
          value="1,284"
          change="+12.5%"
          positive
        />

        <StatCard
          icon={<Clock3 size={22} />}
          title="Avg. Response Time"
          value="2.4h"
          change="-18.2%"
          positive
        />

        <StatCard
          icon={<CheckCircle2 size={22} />}
          title="Resolved"
          value="1,102"
          change="+8.7%"
          positive
        />

        <StatCard
          icon={<AlertTriangle size={22} />}
          title="High Priority"
          value="73"
          change="+4.1%"
          negative
        />
      </section>

      {/* ANALYTICS */}

      <section className="analytics-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3>Tickets by Category</h3>
              <p>Distribution of incoming support requests</p>
            </div>
          </div>

          <div className="bar-chart">
            <ChartBar label="Technical Issue" value={72} />
            <ChartBar label="Billing" value={55} />
            <ChartBar label="Account" value={43} />
            <ChartBar label="Product" value={36} />
            <ChartBar label="Other" value={24} />
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h3>Priority Distribution</h3>
              <p>Current ticket priority levels</p>
            </div>
          </div>

          <div className="priority-chart">
            <div className="donut">
              <div className="donut-center">
                <strong>1,284</strong>
                <span>Tickets</span>
              </div>
            </div>

            <div className="legend">
              <LegendItem label="Low" value="52%" />
              <LegendItem label="Medium" value="31%" />
              <LegendItem label="High" value="12%" />
              <LegendItem label="Critical" value="5%" />
            </div>
          </div>
        </div>
      </section>

      {/* LIVE TRIAGE */}

      <section className="panel triage-panel">
        <div className="panel-header">
          <div>
            <div className="title-with-icon">
              <Bot size={22} />
              <h3>Live AI Ticket Triage</h3>
            </div>

            <p>
              Paste a customer support ticket below and let the AI classify it.
            </p>
          </div>

          <span className="ai-badge">
            <Sparkles size={14} />
            AI Powered
          </span>
        </div>

        <textarea
          className="ticket-input"
          placeholder="Example: I can't log into my account because the password reset email never arrives..."
          value={ticketText}
          onChange={(e) => setTicketText(e.target.value)}
        />

        <div className="triage-actions">
          <span className="character-count">
            {ticketText.length} characters
          </span>

          <button
            className="primary-button"
            onClick={handleTriage}
            disabled={loading || !ticketText.trim()}
          >
            {loading ? (
              <>
                <RefreshCw size={18} className="spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Analyze Ticket
              </>
            )}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {triageResult && <TriageResult result={triageResult} />}
      </section>
    </>
  );
}

function StatCard({ icon, title, value, change, positive, negative }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>

      <div className="stat-content">
        <span className="stat-title">{title}</span>
        <strong className="stat-value">{value}</strong>

        <span
          className={`stat-change ${
            positive ? "positive" : negative ? "negative" : ""
          }`}
        >
          {change} <span>vs. last month</span>
        </span>
      </div>
    </div>
  );
}

function ChartBar({ label, value }) {
  return (
    <div className="chart-row">
      <div className="chart-label">
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>

      <div className="bar-background">
        <div
          className="bar-fill"
          style={{ width: `${value}%` }}
        ></div>
      </div>
    </div>
  );
}

function LegendItem({ label, value }) {
  return (
    <div className="legend-item">
      <div className="legend-label">
        <span className={`legend-dot ${label.toLowerCase()}`}></span>
        {label}
      </div>

      <strong>{value}</strong>
    </div>
  );
}

function TriageResult({ result }) {
  const category =
    result.category ||
    result.predicted_category ||
    result.classification ||
    "Unknown";

  const priority =
    result.priority ||
    result.predicted_priority ||
    "Unknown";

  const team =
    result.team ||
    result.assigned_team ||
    result.department ||
    result.predicted_queue ||
    "Support Team";

  const confidence =
    result.confidence !== undefined
      ? result.confidence
      : result.queue_confidence !== undefined
      ? result.queue_confidence
      : result.score !== undefined
      ? result.score
      : null;

  return (
    <div className="triage-result">
      <div className="result-header">
        <div>
          <span className="result-label">AI TRIAGE RESULT</span>
          <h4>Ticket successfully analyzed</h4>
        </div>

        <CheckCircle2 size={24} />
      </div>

      <div className="result-grid">
        <ResultItem
          label="Category"
          value={category}
        />

        <ResultItem
          label="Priority"
          value={priority}
          badge
        />

        <ResultItem
          label="Assigned Team"
          value={team}
        />

        <ResultItem
          label="Confidence"
          value={
            confidence !== null
              ? `${Math.round(Number(confidence) * 100)}%`
              : "N/A"
          }
        />
      </div>
    </div>
  );
}

function ResultItem({ label, value, badge }) {
  return (
    <div className="result-item">
      <span>{label}</span>

      {badge ? (
        <span className={`priority-badge ${String(value).toLowerCase()}`}>
          {value}
        </span>
      ) : (
        <strong>{value}</strong>
      )}
    </div>
  );
}

/* =========================================================
   CHAT
========================================================= */

function Chat() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! I'm your Support Intelligence assistant. Ask me about support tickets, trends, categories, priorities, or customer issues.",
      citations: [],
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!input.trim() || loading) return;

    const userInput = input;

    const userMessage = {
      role: "user",
      content: userInput,
      citations: [],
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: "demo-session",
          messages: [
            {
              role: "user",
              content: userInput,
            },
          ],
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Server returned ${response.status}: ${errorText}`
        );
      }

      const data = await response.json();

      const answer =
        data.answer ||
        data.response ||
        data.message ||
        data.content ||
        "I received your request, but I couldn't generate an answer.";

      let citations = data.citations || data.sources || [];

      if (
        (!citations || citations.length === 0) &&
        data.cited_ticket_ids
      ) {
        citations = data.cited_ticket_ids
          .split(",")
          .map((id) => `ticket #${id.trim()}`)
          .filter((id) => id !== "ticket #");
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer,
          citations: normalizeCitations(citations),
        },
      ]);
    } catch (err) {
      console.error("Chat error:", err);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I couldn't connect to the backend. Please make sure your FastAPI server is running on port 8000.",
          citations: [],
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="chat-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">SUPPORT INTELLIGENCE</p>
          <h2>AI Support Assistant</h2>
          <p className="page-description">
            Ask questions about your support data and get evidence-backed
            answers.
          </p>
        </div>

        <div className="header-badge">
          <div className="status-dot"></div>
          AI Online
        </div>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.map((message, index) => (
            <Message
              key={index}
              message={message}
            />
          ))}

          {loading && (
            <div className="message assistant-message">
              <div className="avatar assistant-avatar">
                <Bot size={18} />
              </div>

              <div className="message-body">
                <div className="message-name">Support AI</div>

                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="chat-input-area">
          <div className="chat-input-wrapper">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask something about your support tickets..."
              rows={1}
            />

            <button
              className="send-button"
              onClick={sendMessage}
              disabled={!input.trim() || loading}
            >
              <Send size={19} />
            </button>
          </div>

          <p className="chat-hint">
            Press Enter to send • Shift + Enter for a new line
          </p>
        </div>
      </div>
    </div>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";

  return (
    <div
      className={`message ${
        isUser ? "user-message" : "assistant-message"
      }`}
    >
      <div
        className={`avatar ${
          isUser ? "user-avatar" : "assistant-avatar"
        }`}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      <div className="message-body">
        <div className="message-name">
          {isUser ? "You" : "Support AI"}
        </div>

        <div
          className={`message-content ${
            message.error ? "message-error" : ""
          }`}
        >
          {message.content}
        </div>

        {!isUser &&
          message.citations &&
          message.citations.length > 0 && (
            <div className="citations">
              <span className="citation-title">Sources:</span>

              {message.citations.map((citation, index) => (
                <span className="citation" key={index}>
                  {formatCitation(citation)}
                </span>
              ))}
            </div>
          )}
      </div>
    </div>
  );
}

function normalizeCitations(citations) {
  if (!Array.isArray(citations)) {
    return [citations];
  }

  return citations;
}

function formatCitation(citation) {
  if (typeof citation === "string") {
    return citation.startsWith("Source:")
      ? citation
      : `Source: ${citation}`;
  }

  if (typeof citation === "object" && citation !== null) {
    if (citation.source) {
      return `Source: ${citation.source}`;
    }

    if (citation.ticket_id) {
      return `Source: ticket #${citation.ticket_id}`;
    }

    if (citation.id) {
      return `Source: ticket #${citation.id}`;
    }

    if (citation.title) {
      return `Source: ${citation.title}`;
    }
  }

  return `Source: ${String(citation)}`;
}

export default App;

