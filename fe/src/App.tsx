import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ChatArea } from './components/ChatArea';
import { ChatInput } from './components/ChatInput';
import { useChatStorage } from './hooks/useChatStorage';
import { sendRAGQuery, checkServerHealth } from './services/api';
import { Message } from './types/chat';

export const App: React.FC = () => {
  const {
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
  } = useChatStorage();

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [loadingSessionIds, setLoadingSessionIds] = useState<Record<string, boolean>>({});
  const [serverOnline, setServerOnline] = useState(true);

  // Trạng thái loading chỉ tính riêng cho session đang mở hiện tại
  const isCurrentSessionLoading = Boolean(
    currentSession?.id && loadingSessionIds[currentSession.id]
  );

  // Check health periodically
  useEffect(() => {
    const ping = async () => {
      const ok = await checkServerHealth();
      setServerOnline(ok);
    };
    ping();
    const interval = setInterval(ping, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (queryText: string) => {
    if (!currentSession) return;

    const targetSessionId = currentSession.id;
    const targetRAGEnabled = currentSession.rag_enabled;

    const userMessage: Message = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: queryText,
      timestamp: new Date().toISOString(),
    };

    addMessage(targetSessionId, userMessage);
    setLoadingSessionIds((prev) => ({ ...prev, [targetSessionId]: true }));

    try {
      const res = await sendRAGQuery(
        queryText,
        targetRAGEnabled,
        undefined,
        targetSessionId
      );

      const assistantMessage: Message = {
        id: res.request_id || 'msg_' + Date.now(),
        role: 'assistant',
        content: res.answer,
        timestamp: res.timestamp || new Date().toISOString(),
        rag_used: res.rag_used,
        retrieval_mode: res.retrieval_mode,
        rewritten_query: res.rewritten_query,
        nli_verification: res.nli_verification,
        retrieved_chunks: res.retrieved_chunks,
        latency_ms: res.latency_ms,
        input_sha256: res.input_sha256,
      };

      addMessage(targetSessionId, assistantMessage);
    } catch (error: any) {
      const errorMessage: Message = {
        id: 'err_' + Date.now(),
        role: 'assistant',
        content: `⚠️ Không thể kết nối tới máy chủ RAG: ${error.message || error}. Hãy đảm bảo backend FastAPI đang chạy trên cổng 8002.`,
        timestamp: new Date().toISOString(),
        rag_used: targetRAGEnabled,
      };
      addMessage(targetSessionId, errorMessage);
    } finally {
      setLoadingSessionIds((prev) => {
        const next = { ...prev };
        delete next[targetSessionId];
        return next;
      });
    }
  };

  const handleToggleRAG = () => {
    if (currentSession) {
      setRAGEnabled(currentSession.id, !currentSession.rag_enabled);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-chatBg text-gray-100 font-sans">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        isOpen={isSidebarOpen}
        serverOnline={serverOnline}
        loadingSessionIds={loadingSessionIds}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onSelectSession={setActiveSessionId}
        onCreateSession={() => createSession(true)}
        onDeleteSession={deleteSession}
        onRenameSession={renameSession}
      />

      {/* Main Chat Interface */}
      <main className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative">
        <Header
          title={currentSession?.title || 'Đoạn chat mới'}
          ragEnabled={currentSession?.rag_enabled ?? true}
          isLoading={isCurrentSessionLoading}
          onToggleRAG={handleToggleRAG}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          onClearChat={clearCurrentMessages}
        />

        <ChatArea
          messages={currentSession?.messages || []}
          isLoading={isCurrentSessionLoading}
          ragEnabled={currentSession?.rag_enabled ?? true}
          onSendSuggestion={handleSendMessage}
        />

        <ChatInput
          isLoading={isCurrentSessionLoading}
          ragEnabled={currentSession?.rag_enabled ?? true}
          onSendMessage={handleSendMessage}
          onToggleRAG={handleToggleRAG}
        />
      </main>
    </div>
  );
};

export default App;
