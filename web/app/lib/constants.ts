// Keep the browser and the documented uvicorn command on the same local port.
// NEXT_PUBLIC_API_BASE remains available for a separately hosted API.
export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8918"
).replace(/\/+$/, "");
