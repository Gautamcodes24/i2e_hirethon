import { useState, useEffect } from "react";
import {
  PanelLeft,
  PanelLeftClose,
  Settings as SettingsIcon,
  Database,
} from "lucide-react";
import { getHealth } from "./api";
import { useChat } from "./hooks/useChat";
import { useToast } from "./hooks/useToast";
import Sidebar from "./components/Sidebar";
import ChatView from "./components/ChatView";
import SettingsPanel from "./components/SettingsPanel";
import IngestPanel from "./components/IngestPanel";
import ToastContainer from "./components/ToastContainer";

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [ingestOpen, setIngestOpen] = useState(false);
  const [health, setHealth] = useState(null);
  const { toasts, addToast, removeToast } = useToast();

  const {
    chats,
    activeChat,
    activeChatId,
    isStreaming,
    createChat,
    deleteChat,
    selectChat,
    sendMessage,
    stopStreaming,
  } = useChat();

  // Health check
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({ status: "error", index_loaded: false, n_chunks: 0 }),
      );
  }, []);

  const handleNewChat = () => {
    createChat();
    setMobileSidebarOpen(false);
  };

  const handleSelectChat = (id) => {
    selectChat(id);
    setMobileSidebarOpen(false);
  };

  const isMobile = typeof window !== "undefined" && window.innerWidth <= 768;

  const toggleSidebar = () => {
    if (isMobile) {
      setMobileSidebarOpen((v) => !v);
    } else {
      setSidebarCollapsed((v) => !v);
    }
  };

  // Build sidebar className
  const sidebarClass = [
    sidebarCollapsed ? "collapsed" : "",
    mobileSidebarOpen ? "open" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const currentTitle = activeChat?.title || "New Chat";

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onDeleteChat={deleteChat}
        onShowIngest={() => {
          setIngestOpen(true);
          setMobileSidebarOpen(false);
        }}
        onShowSettings={() => {
          setSettingsOpen(true);
          setMobileSidebarOpen(false);
        }}
        className={sidebarClass}
      />

      {/* Mobile overlay */}
      {mobileSidebarOpen && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.5)",
            zIndex: 40,
          }}
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="main-content">
        {/* Top bar */}
        <div className="top-bar">
          <div className="top-bar-left">
            <button
              className="btn-toggle-sidebar btn-icon"
              onClick={toggleSidebar}
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {sidebarCollapsed ? (
                <PanelLeft size={20} />
              ) : (
                <PanelLeftClose size={20} />
              )}
            </button>
            <span className="top-bar-title">{currentTitle}</span>
          </div>
          <div className="top-bar-right">
            <div
              className={`status-pill ${health?.index_loaded ? "" : "offline"}`}
            >
              <span className="status-dot" />
              {health?.index_loaded ? `${health.n_chunks} chunks` : "No index"}
            </div>
            <button
              className="btn-icon"
              onClick={() => setIngestOpen(true)}
              title="Ingest PDF"
            >
              <Database size={18} />
            </button>
            <button
              className="btn-icon"
              onClick={() => setSettingsOpen(true)}
              title="Settings"
            >
              <SettingsIcon size={18} />
            </button>
          </div>
        </div>

        {/* Chat view */}
        <ChatView
          activeChat={activeChat}
          isStreaming={isStreaming}
          onSendMessage={sendMessage}
          onStopStreaming={stopStreaming}
          onQuickPrompt={(q) => sendMessage(q)}
        />
      </div>

      {/* Modals */}
      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onToast={addToast}
      />
      <IngestPanel
        isOpen={ingestOpen}
        onClose={() => setIngestOpen(false)}
        onToast={addToast}
      />
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </div>
  );
}
