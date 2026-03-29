const API = "/api";

// ── Helpers ──────────────────────────────────────────────────────────────────

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// ── Health ───────────────────────────────────────────────────────────────────

export const getHealth = () => request("/health");

// ── Query (non-streaming) ────────────────────────────────────────────────────

export function queryDocument(
  question,
  { topK = 10, alpha = 0.7, useParent = true } = {},
) {
  return request("/query", {
    method: "POST",
    body: JSON.stringify({
      question,
      top_k: topK,
      alpha,
      use_parent: useParent,
    }),
  });
}

// ── Query (SSE streaming) ────────────────────────────────────────────────────

export async function queryStream(
  question,
  { topK = 10, alpha = 0.7, useParent = true } = {},
  onEvent,
) {
  const res = await fetch(`${API}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: topK,
      alpha,
      use_parent: useParent,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("data: ")) {
        try {
          const event = JSON.parse(trimmed.slice(6));
          onEvent(event);
        } catch {
          // skip malformed events
        }
      }
    }
  }
}

// ── Ingestion ────────────────────────────────────────────────────────────────

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API}/ingest`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export const getIngestStatus = (taskId) => request(`/ingest/${taskId}`);
export const getIngestResult = (taskId) => request(`/ingest/${taskId}/result`);

// ── Settings ─────────────────────────────────────────────────────────────────

export const getSettings = () => request("/settings");

export function updateSettings(settings) {
  return request("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}
