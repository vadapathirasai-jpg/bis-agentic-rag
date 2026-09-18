/**
 * BIS Sahayak API Service Layer
 * Connects frontend to FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Health check failed:', err);
    return null;
  }
}

export async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Status check failed:', err);
    return { api: false, qdrant: false, gemini_configured: false };
  }
}

export async function sendChatMessage(userQuestion) {
  const questionText = typeof userQuestion === 'string'
    ? userQuestion
    : (userQuestion?.message || userQuestion?.question || '');
  const trimmed = questionText.trim();
  if (!trimmed) {
    throw new Error('Message cannot be empty');
  }

  // Exact request payload expected by backend POST /api/chat
  const payload = {
    message: trimmed,
  };

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorDetail = 'Unable to process the request right now.';
    try {
      const errorJson = await response.json();
      if (errorJson.message) {
        errorDetail = errorJson.message;
      }
    } catch {
      // ignore parse error
    }
    throw new Error(errorDetail);
  }

  return await response.json();
}
