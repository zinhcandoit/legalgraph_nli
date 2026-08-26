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
  top_k?: number,
  sessionId?: string
): Promise<RAGApiResponse> {
  const payload: Record<string, any> = {
    query,
    rag,
    session_id: sessionId,
  };
  if (typeof top_k === 'number') {
    payload.top_k = top_k;
  }

  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Lỗi từ máy chủ (${response.status})`);
  }

  return response.json();
}