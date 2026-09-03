export interface NLIVerification {
  is_valid: boolean;
  label: string;
  confidence: number;
  probabilities?: Record<string, number>;
  note?: string;
}

export interface RetrievedChunk {
  id?: string;
  text: string;
  score?: number;
  source_type?: string;
  metadata?: Record<string, any>;
  nli_verification?: NLIVerification;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  rag_used?: boolean;
  retrieval_mode?: string;
  rewritten_query?: string;
  nli_verification?: NLIVerification;
  retrieved_chunks?: RetrievedChunk[];
  latency_ms?: number;
  input_sha256?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
  rag_enabled: boolean;
}

export interface RAGApiResponse {
  request_id: string;
  timestamp: string;
  query: string;
  rewritten_query?: string;
  rag_used: boolean;
  retrieval_mode?: string;
  answer: string;
  nli_verification?: NLIVerification;
  retrieved_chunks: RetrievedChunk[];
  total_chunks: number;
  latency_ms: number;
  input_sha256: string;
  session_id?: string;
  metadata?: Record<string, any>;
}
