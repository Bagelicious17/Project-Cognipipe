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
 * POST /api/v1/pipeline/generate
 * Uploads a file and returns the full PipelineResponse JSON.
 */
export async function generatePipeline(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/api/v1/pipeline/generate`, {
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
