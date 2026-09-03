import React, { useState, useRef, useEffect } from 'react';
import { 
  User, 
  Scale, 
  Database, 
  Zap, 
  Copy, 
  Check, 
  ChevronDown, 
  ChevronUp, 
  Sparkles, 
  Clock, 
  BookOpen,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert
} from 'lucide-react';
import { Message, RetrievedChunk } from '../types/chat';
import { ChunkModal } from './ChunkModal';

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  ragEnabled: boolean;
  onSendSuggestion: (prompt: string) => void;
}

const SAMPLE_PROMPTS = [
  'Quy định về thời giờ làm việc và nghỉ ngơi theo Bộ luật Lao động?',
  'Điều kiện có hiệu lực của giao dịch dân sự theo Bộ luật Dân sự?',
  'Các trường hợp loại trừ trách nhiệm hình sự theo Bộ luật Hình sự?',
  'Hạn mức giao đất và cấp Giấy chứng nhận quyền sử dụng đất theo quy định pháp luật?',
];

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isLoading,
  ragEnabled,
  onSendSuggestion,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedChunks, setExpandedChunks] = useState<Record<string, boolean>>({});
  const [selectedChunk, setSelectedChunk] = useState<RetrievedChunk | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleCopy = (content: string, id: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleChunks = (msgId: string) => {
    setExpandedChunks((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto p-4 sm:p-8 flex flex-col items-center justify-center text-center">
        <div className="w-14 h-14 rounded-2xl bg-accentGreen/20 border border-accentGreen/30 text-accentGreen flex items-center justify-center mb-4 shadow-lg">
          <Scale className="w-7 h-7" />
        </div>
        <h2 className="text-xl font-bold text-gray-100 mb-2">
          Trợ lý AI Pháp Luật
        </h2>
        <p className="text-gray-400 text-sm max-w-md mb-8">
          Hệ thống kết hợp <strong>GraphRAG</strong>, <strong>BGE-M3</strong>, <strong>Gemini</strong> và mô hình <strong>BamiBERT NLI</strong> để kiểm chứng tính xác thực logic của câu trả lời.
        </p>

        <div className="w-full max-w-2xl grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {SAMPLE_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => onSendSuggestion(prompt)}
              className="text-left p-3.5 rounded-xl bg-cardBg hover:bg-hoverBg border border-borderDark/70 text-gray-300 hover:text-white text-xs leading-relaxed transition-all duration-150 flex items-start gap-2.5 group"
            >
              <Sparkles className="w-4 h-4 text-accentGreen flex-shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
              <span>{prompt}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
      {messages.map((msg) => {
        const isUser = msg.role === 'user';
        const nli = msg.nli_verification;

        return (
          <div
            key={msg.id}
            className={`flex gap-3.5 max-w-4xl mx-auto ${
              isUser ? 'justify-end' : 'justify-start'
            }`}
          >
            {!isUser && (
              <div className="w-8 h-8 rounded-xl bg-accentGreen/20 border border-accentGreen/30 text-accentGreen flex items-center justify-center flex-shrink-0 mt-1">
                <Scale className="w-4 h-4" />
              </div>
            )}

            <div
              className={`flex flex-col space-y-2.5 max-w-[85%] sm:max-w-[78%] ${
                isUser
                  ? 'bg-accentGreen/15 border border-accentGreen/30 text-gray-100 rounded-2xl rounded-tr-sm px-4 py-3'
                  : 'bg-cardBg border border-borderDark/60 text-gray-100 rounded-2xl rounded-tl-sm px-4 py-3.5 shadow-sm'
              }`}
            >
              {/* Header Badges for Assistant */}
              {!isUser && (
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-borderDark/40 text-[11px]">
                  <div className="flex items-center gap-2">
                    {msg.rag_used ? (
                      <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-medium flex items-center gap-1">
                        <Database className="w-3 h-3" />
                        RAG Graph DB
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30 font-medium flex items-center gap-1">
                        <Zap className="w-3 h-3" />
                        Direct LLM
                      </span>
                    )}

                    {msg.latency_ms && (
                      <span className="text-gray-400 flex items-center gap-1">
                        <Clock className="w-3 h-3 text-gray-500" />
                        {msg.latency_ms}ms
                      </span>
                    )}
                  </div>

                  {/* NLI Verification Status Badge */}
                  {nli && (
                    <div className="flex items-center gap-1.5">
                      {nli.is_valid ? (
                        <span
                          className="px-2.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 font-semibold flex items-center gap-1 text-[11px] shadow-sm"
                          title={`Xác thực NLI: HỢP LỆ (Entailment) - Độ tin cậy: ${(nli.confidence * 100).toFixed(1)}%`}
                        >
                          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                          NLI: ĐẠT ({(nli.confidence * 100).toFixed(0)}%)
                        </span>
                      ) : (
                        <span
                          className="px-2.5 py-0.5 rounded-full bg-rose-500/15 border border-rose-500/40 text-rose-300 font-semibold flex items-center gap-1 text-[11px] shadow-sm"
                          title={`Xác thực NLI: KHÔNG ĐẠT (Contradiction / Nghi ngờ sai lệch) - Độ tin cậy: ${(nli.confidence * 100).toFixed(1)}%`}
                        >
                          <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                          NLI: CẢNH BÁO ({(nli.confidence * 100).toFixed(0)}%)
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Rewritten Query notification if RAG applied */}
              {!isUser && msg.rewritten_query && (
                <div className="text-[11px] text-gray-400 bg-chatBg/80 rounded-lg p-2 border border-borderDark/40">
                  <span className="font-medium text-accentGreen">Truy vấn pháp lý chuẩn hóa:</span>{' '}
                  <span className="italic">{msg.rewritten_query}</span>
                </div>
              )}

              {/* Message Content */}
              <div className="text-sm leading-relaxed whitespace-pre-wrap select-text">
                {msg.content}
              </div>

              {/* NLI Note Box */}
              {!isUser && nli && (
                <div className={`p-2.5 rounded-xl border text-xs flex items-start gap-2 ${
                  nli.is_valid 
                    ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-200' 
                    : 'bg-rose-950/20 border-rose-500/30 text-rose-200'
                }`}>
                  {nli.is_valid ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <span className="font-semibold">{nli.is_valid ? 'Chứng thực bởi Luật:' : 'Cảnh báo đối chiếu Luật:'}</span>{' '}
                    <span>{nli.note}</span>
                  </div>
                </div>
              )}

              {/* Retrieved Chunks Accordion */}
              {!isUser && msg.retrieved_chunks && msg.retrieved_chunks.length > 0 && (
                <div className="mt-2 pt-2 border-t border-borderDark/50">
                  <button
                    onClick={() => toggleChunks(msg.id)}
                    className="flex items-center justify-between w-full text-xs font-medium text-accentGreen hover:text-emerald-300 py-1"
                  >
                    <span className="flex items-center gap-1.5">
                      <BookOpen className="w-3.5 h-3.5" />
                      {msg.retrieved_chunks.length} trích đoạn căn cứ pháp lý được tham chiếu
                    </span>
                    {expandedChunks[msg.id] ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>

                  {expandedChunks[msg.id] && (
                    <div className="mt-2 space-y-2">
                      {msg.retrieved_chunks.map((chunk, cIdx) => (
                        <div
                          key={cIdx}
                          onClick={() => setSelectedChunk(chunk)}
                          className="p-2.5 rounded-lg bg-chatBg hover:bg-hoverBg border border-borderDark/70 cursor-pointer text-xs transition-colors"
                        >
                          <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1 gap-2 flex-wrap">
                            <span className="font-semibold text-gray-300 flex items-center gap-1.5">
                              <span>Trích đoạn #{cIdx + 1}</span>
                              <span className="text-gray-500 font-normal">({chunk.source_type || 'text_unit'})</span>
                            </span>
                            <div className="flex items-center gap-2">
                              {chunk.nli_verification && (
                                <span
                                  className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                                    chunk.nli_verification.is_valid
                                      ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                      : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                                  }`}
                                  title={`NLI: ${chunk.nli_verification.label} (${(chunk.nli_verification.confidence * 100).toFixed(1)}%) - ${chunk.nli_verification.note || ''}`}
                                >
                                  {chunk.nli_verification.is_valid ? '✓ NLI: Hợp lệ' : '✗ NLI: Cảnh báo'} ({(chunk.nli_verification.confidence * 100).toFixed(0)}%)
                                </span>
                              )}
                              {chunk.score !== undefined && (
                                <span className="text-emerald-400">Score: {chunk.score.toFixed(3)}</span>
                              )}
                            </div>
                          </div>
                          <p className="text-gray-300 line-clamp-2 italic">
                            "{chunk.text}"
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Bottom Actions */}
              <div className="flex items-center justify-between pt-1 text-[11px] text-gray-500">
                <span>{new Date(msg.timestamp).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}</span>
                <button
                  onClick={() => handleCopy(msg.content, msg.id)}
                  className="p-1 text-gray-400 hover:text-white rounded transition-colors"
                  title="Sao chép câu trả lời"
                >
                  {copiedId === msg.id ? (
                    <Check className="w-3.5 h-3.5 text-accentGreen" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
            </div>

            {isUser && (
              <div className="w-8 h-8 rounded-xl bg-gray-700 border border-gray-600 text-gray-300 flex items-center justify-center flex-shrink-0 mt-1">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        );
      })}

      {/* Loading indicator */}
      {isLoading && (
        <div className="flex gap-3.5 max-w-4xl mx-auto items-start">
          <div
            className={`w-8 h-8 rounded-xl border flex items-center justify-center flex-shrink-0 mt-1 ${
              ragEnabled
                ? 'bg-accentGreen/20 border-accentGreen/30 text-accentGreen'
                : 'bg-amber-500/20 border-amber-500/30 text-amber-400'
            }`}
          >
            {ragEnabled ? (
              <Scale className="w-4 h-4 animate-pulse" />
            ) : (
              <Zap className="w-4 h-4 animate-pulse" />
            )}
          </div>
          <div className="bg-cardBg border border-borderDark/60 rounded-2xl rounded-tl-sm px-4 py-3 text-xs text-gray-300 flex items-center gap-2.5 shadow-sm">
            <span
              className={`inline-block w-2 h-2 rounded-full animate-ping flex-shrink-0 ${
                ragEnabled ? 'bg-accentGreen' : 'bg-amber-400'
              }`}
            />
            <span>
              {ragEnabled
                ? 'Đang tra cứu đồ thị pháp luật, tạo câu trả lời và chạy kiểm định NLI...'
                : 'Đang gửi truy vấn trực tiếp tới mô hình LLM (chế độ Direct LLM)...'}
            </span>
          </div>
        </div>
      )}

      <div ref={bottomRef} />

      {/* Modal xem chi tiết chunk */}
      <ChunkModal chunk={selectedChunk} onClose={() => setSelectedChunk(null)} />
    </div>
  );
};
