import {
  MessageSquare,
  Plus,
  Trash2,
  Upload,
  Settings,
  BookOpen,
} from "lucide-react";

export default function Sidebar({
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onShowIngest,
  onShowSettings,
  className = "",
}) {
  // Group chats: today, previous 7 days, older
  const now = Date.now();
  const dayMs = 86400000;
  const todayChats = chats.filter((c) => now - c.createdAt < dayMs);
  const weekChats = chats.filter(
    (c) => now - c.createdAt >= dayMs && now - c.createdAt < 7 * dayMs,
  );
  const olderChats = chats.filter((c) => now - c.createdAt >= 7 * dayMs);

  const renderSection = (label, items) => {
    if (items.length === 0) return null;
    return (
      <div key={label}>
        <div className="sidebar-section-label">{label}</div>
        {items.map((chat) => (
          <div
            key={chat.id}
            className={`chat-item ${chat.id === activeChatId ? "active" : ""}`}
            onClick={() => onSelectChat(chat.id)}
          >
            <MessageSquare size={14} className="chat-item-icon" />
            <span className="chat-item-text">{chat.title || "New Chat"}</span>
            <button
              className="chat-item-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteChat(chat.id);
              }}
              title="Delete chat"
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>
    );
  };

  return (
    <aside className={`sidebar ${className}`}>
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <BookOpen size={18} />
          </div>
          <div>
            <h2>Tech Manual QA</h2>
            <div className="sidebar-brand-sub">RAG-Powered Assistant</div>
          </div>
        </div>
        <button className="btn-new-chat" onClick={onNewChat}>
          <Plus size={15} /> New Chat
        </button>
      </div>

      <div className="sidebar-chats">
        {chats.length === 0 && (
          <div
            style={{
              padding: "2rem 1rem",
              textAlign: "center",
              color: "var(--text-muted)",
              fontSize: "0.82rem",
            }}
          >
            No conversations yet.
            <br />
            Start by asking a question!
          </div>
        )}
        {renderSection("Today", todayChats)}
        {renderSection("Previous 7 Days", weekChats)}
        {renderSection("Older", olderChats)}
      </div>

      <div className="sidebar-footer">
        <button className="sidebar-footer-btn" onClick={onShowIngest}>
          <Upload size={16} /> Ingest PDF
        </button>
        <button className="sidebar-footer-btn" onClick={onShowSettings}>
          <Settings size={16} /> Settings
        </button>
      </div>
    </aside>
  );
}
