import { useRef, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Square,
  BookOpen,
  ChevronDown,
  ChevronRight,
  FileText,
  Sparkles,
  ExternalLink,
} from "lucide-react";

const QUICK_PROMPTS = [
  "What is systems engineering?",
  "Explain the V-model lifecycle",
  "What are key NASA SE processes?",
  "Summarize risk management approach",
];

function MessageBubble({ message }) {
  const [showCitations, setShowCitations] = useState(false);

  return (
    <div className={`message ${message.role}`}>
      <div className={`message-avatar ${message.role}`}>
        {message.role === "user" ? "U" : <Sparkles size={16} />}
      </div>
      <div className="message-body">
        <div className="message-sender">
          {message.role === "user" ? "You" : "Assistant"}
        </div>
        <div className={`message-content ${message.error ? "error" : ""}`}>
          {message.role === "assistant" &&
          !message.content &&
          message.streaming ? (
            <div className="loading-dots">
              <span /> <span /> <span />
            </div>
          ) : (
            <>
              <ReactMarkdown>{message.content}</ReactMarkdown>
              {message.streaming && <span className="streaming-cursor" />}
            </>
          )}
        </div>

        {message.citations && message.citations.length > 0 && (
          <div className="citations-card">
            <div
              className="citations-header"
              onClick={() => setShowCitations(!showCitations)}
            >
              {showCitations ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
              <FileText size={13} />
              {message.citations.length} Source
              {message.citations.length !== 1 ? "s" : ""}
            </div>
            {showCitations && (
              <div className="citations-list">
                {message.citations.map((c, i) => {
                  const page = c.page ?? null;
                  const pdfName =
                    c.pdf_name || c.citation?.match(/^Page \d+/)?.[0];
                  const pdfLink =
                    c.pdf_name && page
                      ? `/api/pdf/${encodeURIComponent(c.pdf_name)}#page=${page}`
                      : null;

                  return (
                    <div key={i} className="citation-item">
                      <span className="citation-badge">{i + 1}</span>
                      <div className="citation-detail">
                        {c.pdf_name && (
                          <span className="citation-pdf-name">
                            <FileText size={12} />
                            {c.pdf_name}
                          </span>
                        )}
                        <span className="citation-text">
                          {c.citation || `Source ${i + 1}`}
                        </span>
                      </div>
                      {page && <span className="citation-page">p.{page}</span>}
                      {pdfLink && (
                        <a
                          href={pdfLink}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="citation-link"
                          title={`Open page ${page}`}
                        >
                          <ExternalLink size={13} />
                        </a>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatView({
  activeChat,
  isStreaming,
  onSendMessage,
  onStopStreaming,
  onQuickPrompt,
}) {
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const [input, setInput] = useState("");

  const messages = activeChat?.messages || [];

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, messages[messages.length - 1]?.content]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  const handleSend = () => {
    const q = input.trim();
    if (!q || isStreaming) return;
    setInput("");
    onSendMessage(q);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Empty state
  if (!activeChat || messages.length === 0) {
    return (
      <>
        <div className="chat-area">
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <BookOpen size={28} />
            </div>
            <h3>Ask anything about your documents</h3>
            <p>
              Upload a technical PDF via the Ingest panel, then ask questions
              here. Answers are grounded in the document with citations.
            </p>
            <div className="quick-prompts">
              {QUICK_PROMPTS.map((q) => (
                <button
                  key={q}
                  className="quick-prompt"
                  onClick={() => {
                    setInput(q);
                    if (onQuickPrompt) onQuickPrompt(q);
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="chat-input-container">
          <div className="chat-input-wrapper">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about the document..."
            />
            <button
              className="btn-send"
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
            >
              <Send size={16} />
            </button>
          </div>
          <div className="chat-input-hint">
            Press Enter to send, Shift+Enter for new line
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="chat-area">
        <div className="messages-container">
          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a follow-up question..."
          />
          {isStreaming ? (
            <button
              className="btn-stop"
              onClick={onStopStreaming}
              title="Stop generating"
            >
              <Square size={14} />
            </button>
          ) : (
            <button
              className="btn-send"
              onClick={handleSend}
              disabled={!input.trim()}
            >
              <Send size={16} />
            </button>
          )}
        </div>
        <div className="chat-input-hint">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </>
  );
}
