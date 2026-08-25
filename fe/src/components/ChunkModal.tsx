import React from 'react';
import { X, FileText, Award, Hash } from 'lucide-react';
import { RetrievedChunk } from '../types/chat';

interface ChunkModalProps {
  chunk: RetrievedChunk | null;
  onClose: () => void;
}

export const ChunkModal: React.FC<ChunkModalProps> = ({ chunk, onClose }) => {
  if (!chunk) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-cardBg border border-borderDark w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="px-5 py-4 border-b border-borderDark flex items-center justify-between bg-sidebarBg">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-accentGreen" />
            <h3 className="font-semibold text-gray-200 text-sm">
              Chi tiết trích đoạn pháp lý từ Graph DB
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-hoverBg"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-5 space-y-4 max-h-[70vh] overflow-y-auto text-sm">
          <div className="flex flex-wrap gap-2 text-xs">
            {chunk.score !== undefined && (
              <span className="px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-medium flex items-center gap-1">
                <Award className="w-3.5 h-3.5" />
                Reranker Score: {chunk.score.toFixed(4)}
              </span>
            )}
            <span className="px-2.5 py-1 rounded-md bg-blue-500/10 border border-blue-500/30 text-blue-400 font-medium">
              Nguồn: {chunk.source_type || 'text_unit'}
            </span>
            {chunk.id && (
              <span className="px-2.5 py-1 rounded-md bg-gray-700/50 border border-gray-600 text-gray-300 font-mono flex items-center gap-1">
                <Hash className="w-3.5 h-3.5" />
                {chunk.id.slice(0, 16)}...
              </span>
            )}
          </div>

          <div className="p-4 rounded-xl bg-chatBg border border-borderDark/80 text-gray-200 whitespace-pre-wrap leading-relaxed">
            {chunk.text}
          </div>
        </div>

        <div className="px-5 py-3 border-t border-borderDark bg-sidebarBg flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-hoverBg hover:bg-gray-600 text-white text-xs font-medium transition-colors"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};