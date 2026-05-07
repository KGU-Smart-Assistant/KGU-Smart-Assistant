const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    let detail = `API request failed with status ${response.status}`;

    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // Keep the status-based message when the response body is not JSON.
    }

    throw new Error(detail);
  }

  return response.json();
}

export function sendChatMessage(message) {
  return request("/api/v1/chat/chat", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

export function listContacts() {
  return request("/api/v1/extra/contacts");
}

export function listPlaces() {
  return request("/api/v1/extra/places");
}
