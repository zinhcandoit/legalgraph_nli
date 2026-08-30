import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  isLoading: boolean;
  ragEnabled: boolean;
  onSendMessage: (query: string) => void;
  onToggleRAG?: () => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  isLoading,
  ragEnabled,
  onSendMessage,
}) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="p-3 sm:p-4 bg-chatBg/80 border-t border-borderDark/40">
      <form
        onSubmit={handleSubmit}
        className="max-w-4xl mx-auto relative rounded-2xl bg-inputBg border border-borderDark/80 shadow-lg focus-within:border-accentGreen/70 transition-all"
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Hỏi về các quy định pháp luật (Enter để gửi, Shift+Enter xuống dòng)..."
          rows={1}
          disabled={isLoading}
          className="w-full bg-transparent text-gray-100 text-sm placeholder-gray-500 py-3.5 pl-4 pr-24 resize-none focus:outline-none max-h-48"
        />

        <div className="absolute right-2.5 bottom-2.5 flex items-center gap-1.5">
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className={`p-2 rounded-xl flex items-center justify-center transition-all ${
              input.trim() && !isLoading
                ? 'bg-accentGreen hover:bg-accentGreenHover text-white shadow-md'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
            title="Gửi câu hỏi"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>

      <div className="max-w-4xl mx-auto mt-2 flex items-center justify-between text-[11px] text-gray-500 px-2">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1">
            Chế độ:
            <strong className={ragEnabled ? 'text-emerald-400' : 'text-amber-400'}>
              {ragEnabled ? '🌿 RAG Đồ thị Tri thức' : '⚡ Trả lời trực tiếp'}
            </strong>
          </span>
        </div>
        <span>VietLegal AI • BamiBERT ViLegalNLI</span>
      </div>
    </div>
  );
};