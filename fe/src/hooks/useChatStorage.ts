import { useState, useEffect } from 'react';
import { ChatSession, Message } from '../types/chat';

const STORAGE_KEY = 'vietlegal_chat_sessions_v1';
const ACTIVE_SESSION_KEY = 'vietlegal_active_session_id';

function createNewSession(rag_enabled = true): ChatSession {
  const now = new Date().toISOString();
  return {
    id: 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
    title: 'Đoạn chat mới',
    messages: [],
    created_at: now,
    updated_at: now,
    rag_enabled,
  };
}

export function useChatStorage() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.error('Lỗi đọc chat history từ localStorage:', e);
    }
    return [createNewSession(true)];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    const savedId = localStorage.getItem(ACTIVE_SESSION_KEY);
    if (savedId && sessions.some((s) => s.id === savedId)) {
      return savedId;
    }
    return sessions[0]?.id || '';
  });

  // Sync to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (e) {
      console.error('Lỗi lưu sessions vào localStorage:', e);
    }
  }, [sessions]);

  useEffect(() => {
    localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
  }, [activeSessionId]);

  const currentSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  const createSession = (rag_enabled = true) => {
    const newSess = createNewSession(rag_enabled);
    setSessions((prev) => [newSess, ...prev]);
    setActiveSessionId(newSess.id);
    return newSess;
  };

  const deleteSession = (id: string) => {
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== id);
      if (filtered.length === 0) {
        const fresh = createNewSession(true);
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (id === activeSessionId) {
        setActiveSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  const renameSession = (id: string, newTitle: string) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, title: newTitle, updated_at: new Date().toISOString() } : s))
    );
  };

  const setRAGEnabled = (id: string, enabled: boolean) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, rag_enabled: enabled, updated_at: new Date().toISOString() } : s))
    );
  };

  const addMessage = (sessionId: string, message: Message) => {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s;

        const updatedMessages = [...s.messages, message];
        let title = s.title;

        // Tự động đặt tên theo câu hỏi đầu tiên của người dùng
        if (s.title === 'Đoạn chat mới' && message.role === 'user') {
          title = message.content.slice(0, 32).trim() + (message.content.length > 32 ? '...' : '');
        }

        return {
          ...s,
          title,
          messages: updatedMessages,
          updated_at: new Date().toISOString(),
        };
      })
    );
  };

  const clearCurrentMessages = () => {
    if (!currentSession) return;
    setSessions((prev) =>
      prev.map((s) => (s.id === currentSession.id ? { ...s, messages: [], updated_at: new Date().toISOString() } : s))
    );
  };

  return {
    sessions,
    currentSession,
    activeSessionId,
    setActiveSessionId,
    createSession,
    deleteSession,
    renameSession,
    setRAGEnabled,
    addMessage,
    clearCurrentMessages,
  };
}