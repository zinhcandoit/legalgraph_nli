import React from 'react';
import { Menu, Zap, Database, Trash2 } from 'lucide-react';

interface HeaderProps {
  title: string;
  ragEnabled: boolean;
  isLoading?: boolean;
  onToggleRAG: () => void;
  onToggleSidebar: () => void;
  onClearChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  ragEnabled,
  isLoading = false,
  onToggleRAG,
  onToggleSidebar,
  onClearChat,
}) => {
  return (
    <header className="h-14 border-b border-borderDark/60 bg-sidebarBg/40 backdrop-blur px-4 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg text-gray-400 hover:bg-hoverBg hover:text-white transition-colors"
          title="Ẩn / Hiện Sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="flex flex-col">
          <h1 className="text-sm font-semibold text-gray-200 truncate max-w-[180px] sm:max-w-xs md:max-w-md">
            {title}
          </h1>
          <span className="text-[10px] text-gray-400 hidden sm:inline">
            Tra cứu Pháp Luật • Microsoft GraphRAG & BGE-M3
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* RAG Switch Button */}
        <div
          className={`flex items-center gap-2 bg-cardBg/90 border border-borderDark/80 rounded-full px-3 py-1 shadow-sm transition-all duration-200 ${
            isLoading ? 'opacity-50 cursor-not-allowed pointer-events-none' : ''
          }`}
        >
          <div className="flex items-center gap-1.5 text-xs font-medium">
            {ragEnabled ? (
              <span className="flex items-center gap-1 text-emerald-400">
                <Database className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">RAG Đồ thị</span>
              </span>
            ) : (
              <span className="flex items-center gap-1 text-amber-400">
                <Zap className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Direct LLM</span>
              </span>
            )}
          </div>

          <button
            type="button"
            role="switch"
            aria-checked={ragEnabled}
            disabled={isLoading}
            onClick={onToggleRAG}
            className={`relative inline-flex h-5 w-9 flex-shrink-0 rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
              isLoading ? 'cursor-not-allowed' : 'cursor-pointer'
            } ${ragEnabled ? 'bg-accentGreen' : 'bg-gray-600'}`}
            title={
              isLoading
                ? 'Đang gửi truy vấn, không thể đổi chế độ'
                : ragEnabled
                ? 'Đang BẬT RAG (Tra cứu đồ thị luật)'
                : 'Đang TẮT RAG (Hỏi trực tiếp LLM)'
            }
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                ragEnabled ? 'translate-x-4' : 'translate-x-0'
              }`}
            />
          </button>
        </div>

        {/* Clear Chat Button */}
        <button
          onClick={onClearChat}
          className="p-2 text-gray-400 hover:text-red-400 hover:bg-hoverBg rounded-lg transition-colors"
          title="Xóa lịch sử hội thoại hiện tại"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};