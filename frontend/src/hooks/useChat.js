import { useState, useCallback, useRef } from "react";
import { queryStream } from "../api";

const STORAGE_KEY = "techqa-chats";

function loadChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveChats(chats) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

export function useChat() {
  const [chats, setChats] = useState(loadChats);
  const [activeChatId, setActiveChatId] = useState(() => chats[0]?.id || null);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef(null);

  const activeChat = chats.find((c) => c.id === activeChatId) || null;

  const persist = (updated) => {
    setChats(updated);
    saveChats(updated);
  };

  const createChat = useCallback(() => {
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const newChat = {
      id,
      title: "New Chat",
      messages: [],
      createdAt: Date.now(),
    };
    const updated = [newChat, ...chats];
    persist(updated);
    setActiveChatId(id);
    return id;
  }, [chats]);

  const deleteChat = useCallback(
    (id) => {
      const updated = chats.filter((c) => c.id !== id);
      persist(updated);
      if (activeChatId === id) {
        setActiveChatId(updated[0]?.id || null);
      }
    },
    [chats, activeChatId],
  );

  const selectChat = useCallback((id) => {
    setActiveChatId(id);
  }, []);

  const sendMessage = useCallback(
    async (question) => {
      let chatId = activeChatId;

      // Auto-create chat if none
      if (!chatId) {
        chatId =
          Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
        const newChat = {
          id: chatId,
          title: question.slice(0, 50),
          messages: [],
          createdAt: Date.now(),
        };
        const updated = [newChat, ...chats];
        persist(updated);
        setActiveChatId(chatId);
      }

      const userMsg = { role: "user", content: question, ts: Date.now() };
      const assistantMsg = {
        role: "assistant",
        content: "",
        citations: [],
        ts: Date.now(),
        streaming: true,
      };

      setChats((prev) => {
        const updated = prev.map((c) => {
          if (c.id !== chatId) return c;
          const title =
            c.messages.length === 0 ? question.slice(0, 50) : c.title;
          return {
            ...c,
            title,
            messages: [...c.messages, userMsg, assistantMsg],
          };
        });
        saveChats(updated);
        return updated;
      });

      setIsStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await queryStream(question, {}, (event) => {
          if (controller.signal.aborted) return;

          if (event.type === "token") {
            setChats((prev) => {
              const updated = prev.map((c) => {
                if (c.id !== chatId) return c;
                const msgs = [...c.messages];
                const last = { ...msgs[msgs.length - 1] };
                last.content += event.content;
                msgs[msgs.length - 1] = last;
                return { ...c, messages: msgs };
              });
              saveChats(updated);
              return updated;
            });
          } else if (event.type === "citations") {
            setChats((prev) => {
              const updated = prev.map((c) => {
                if (c.id !== chatId) return c;
                const msgs = [...c.messages];
                const last = { ...msgs[msgs.length - 1] };
                last.citations = event.citations || [];
                msgs[msgs.length - 1] = last;
                return { ...c, messages: msgs };
              });
              saveChats(updated);
              return updated;
            });
          } else if (event.type === "done") {
            setChats((prev) => {
              const updated = prev.map((c) => {
                if (c.id !== chatId) return c;
                const msgs = [...c.messages];
                const last = { ...msgs[msgs.length - 1] };
                last.streaming = false;
                msgs[msgs.length - 1] = last;
                return { ...c, messages: msgs };
              });
              saveChats(updated);
              return updated;
            });
          }
        });
      } catch (err) {
        if (!controller.signal.aborted) {
          setChats((prev) => {
            const updated = prev.map((c) => {
              if (c.id !== chatId) return c;
              const msgs = [...c.messages];
              const last = { ...msgs[msgs.length - 1] };
              last.content = last.content || `Error: ${err.message}`;
              last.streaming = false;
              last.error = true;
              msgs[msgs.length - 1] = last;
              return { ...c, messages: msgs };
            });
            saveChats(updated);
            return updated;
          });
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [activeChatId, chats],
  );

  const stopStreaming = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      setIsStreaming(false);
      setChats((prev) => {
        const updated = prev.map((c) => {
          if (c.id !== activeChatId) return c;
          const msgs = [...c.messages];
          if (msgs.length > 0) {
            const last = { ...msgs[msgs.length - 1] };
            last.streaming = false;
            msgs[msgs.length - 1] = last;
          }
          return { ...c, messages: msgs };
        });
        saveChats(updated);
        return updated;
      });
    }
  }, [activeChatId]);

  return {
    chats,
    activeChat,
    activeChatId,
    isStreaming,
    createChat,
    deleteChat,
    selectChat,
    sendMessage,
    stopStreaming,
  };
}
