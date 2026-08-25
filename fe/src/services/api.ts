import { RAGApiResponse } from '../types/chat';

const API_BASE_URL = 'http://localhost:8002';

export async function checkServerHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    return res.ok;
  } catch (error) {
    return false;
  }
}

export async function sendRAGQuery(
  query: string,
  rag: boolean,
  top_k: number = 5,
  sessionId?: string
): Promise<RAGApiResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      rag,
      top_k,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Lỗi từ máy chủ (${response.status})`);
  }

  return response.json();
}