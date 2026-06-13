/**
 * CogniPipe — API Client
 * =======================
 * All backend communication in one place.
 * Uses fetch() exclusively (no axios).
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

/**
 * POST /api/v1/pipeline/profile-only
 * Uploads a file and returns the ProfileResult JSON.
 */
export async function profileDataset(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/api/v1/pipeline/profile-only`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw { status: res.status, ...err };
  }

  return res.json();
}

/**
 * POST /api/v1/pipeline/generate  (NDJSON streaming)
 *
 * Uploads a file and reads the streaming NDJSON response.
 * Calls `onProgress({ progress, message })` for each progress event.
 * Returns the final PipelineResponse when the stream emits a "done" event.
 * Throws on "error" events or HTTP failures.
 *
 * Uses a buffer pattern to handle chunks that may split across
 * JSON line boundaries — never splits raw chunks by '\n' directly.
 */
export async function generatePipeline(file, onProgress) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/api/v1/pipeline/generate`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    // Non-streaming error (validation failures return normal JSON)
    const err = await res.json();
    throw { status: res.status, ...err };
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      // Stream ended unexpectedly without a done/error event
      throw {
        error: "stream_error",
        detail: "Connection to server was lost before the pipeline finished.",
        stage: "orchestration",
      };
    }

    buffer += decoder.decode(value, { stream: true });

    // Process all complete lines in the buffer
    let newlineIdx;
    while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newlineIdx).trim();
      buffer = buffer.slice(newlineIdx + 1);

      if (!line) continue; // skip empty lines

      let event;
      try {
        event = JSON.parse(line);
      } catch {
        continue; // skip malformed lines
      }

      if (event.type === "progress") {
        if (onProgress) {
          onProgress({ progress: event.progress, message: event.message });
        }
      } else if (event.type === "done") {
        return event.data;
      } else if (event.type === "error") {
        throw {
          error: event.error,
          detail: event.detail,
          stage: event.stage,
        };
      }
    }
  }
}

/**
 * POST /api/v1/pipeline/download/{type}
 * Sends content to backend, receives a file blob, and triggers download.
 */
export async function downloadFile(type, content, filename) {
  const res = await fetch(`${API_URL}/api/v1/pipeline/download/${type}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, filename }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw { status: res.status, ...err };
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
