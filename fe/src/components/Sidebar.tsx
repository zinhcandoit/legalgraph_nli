import React, { useState } from 'react';
import { 
  Plus, 
  MessageSquare, 
  Trash2, 
  Edit2, 
  Check, 
  X, 
  Scale, 
  ChevronLeft, 
  ChevronRight,
  Database,
  Activity
} from 'lucide-react';
import { ChatSession } from '../types/chat';

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string;
  isOpen: boolean;
  serverOnline: boolean;
  onToggleSidebar: () => void;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newTitle: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  activeSessionId,
  isOpen,
  serverOnline,
  onToggleSidebar,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onRenameSession,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const handleStartRename = (session: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(session.id);
    setEditTitle(session.title);
  };

  const handleSaveRename = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (editTitle.trim()) {
      onRenameSession(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
  };

  return (
    <aside
      className={`fixed md:static inset-y-0 left-0 z-40 flex flex-col bg-sidebarBg border-r border-borderDark transition-all duration-300 ${
        isOpen ? 'w-64 translate-x-0' : '-translate-x-full md:translate-x-0 md:w-0 md:border-r-0'
      }`}
    >
      {/* Header Logo & New Chat */}
      <div className="p-3.5 flex flex-col gap-3 border-b border-borderDark/40">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-accentGreen/20 flex items-center justify-center text-accentGreen border border-accentGreen/30">
              <Scale className="w-5 h-5" />
            </div>
            <span className="font-semibold text-gray-200 tracking-tight text-sm">
              VietLegal AI
            </span>
          </div>
          <button
            onClick={onToggleSidebar}
            className="md:hidden p-1.5 rounded-lg text-gray-400 hover:bg-hoverBg hover:text-white"
            title="Đóng sidebar"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
        </div>

        <button
          onClick={onCreateSession}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl bg-cardBg hover:bg-hoverBg text-gray-100 text-sm font-medium border border-borderDark/60 shadow-sm transition-colors duration-150"
        >
          <Plus className="w-4 h-4 text-accentGreen" />
          <span>Cuộc trò chuyện mới</span>
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        <div className="px-2 py-1 text-[11px] font-medium tracking-wider text-gray-400 uppercase">
          Lịch sử trò chuyện
        </div>

        {sessions.map((session) => {
          const isActive = session.id === activeSessionId;
          const isEditing = editingId === session.id;

          return (
            <div
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`group relative flex items-center gap-2.5 px-3 py-2.5 rounded-xl cursor-pointer text-sm transition-all duration-150 ${
                isActive
                  ? 'bg-cardBg text-white font-medium border border-borderDark/50 shadow-sm'
                  : 'text-gray-400 hover:bg-cardBg/60 hover:text-gray-200'
              }`}
            >
              <MessageSquare className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-accentGreen' : 'text-gray-500'}`} />

              {isEditing ? (
                <div className="flex-1 flex items-center gap-1">
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    className="w-full bg-inputBg text-xs text-white px-2 py-1 rounded border border-accentGreen focus:outline-none"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSaveRename(session.id, e as any);
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                  />
                  <button
                    onClick={(e) => handleSaveRename(session.id, e)}
                    className="p-1 hover:text-accentGreen"
                  >
                    <Check className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={handleCancelRename} className="p-1 hover:text-red-400">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ) : (
                <>
                  <span className="flex-1 truncate text-xs">{session.title}</span>

                  {/* Actions hover */}
                  <div
                    className={`flex items-center gap-1 ${
                      isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                    } transition-opacity`}
                  >
                    <button
                      onClick={(e) => handleStartRename(session, e)}
                      className="p-1 text-gray-400 hover:text-white rounded"
                      title="Đổi tên"
                    >
                      <Edit2 className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(session.id);
                      }}
                      className="p-1 text-gray-400 hover:text-red-400 rounded"
                      title="Xóa đoạn chat"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* Server Status Footer */}
      <div className="p-3 border-t border-borderDark/40 bg-sidebarBg/50 text-xs text-gray-400 flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-3.5 h-3.5 text-accentGreen" />
            <span className="text-gray-300 font-medium">Graph DB & NLI</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                serverOnline ? 'bg-emerald-500 shadow-[0_0_8px_#10b981]' : 'bg-red-500'
              }`}
            />
            <span className={serverOnline ? 'text-emerald-400 font-medium' : 'text-red-400'}>
              {serverOnline ? 'Sẵn sàng' : 'Mất kết nối'}
            </span>
          </div>
        </div>
        <div className="text-[10px] text-gray-500">
          Backend: FastAPI (Port 8002)
        </div>
      </div>
    </aside>
  );
};