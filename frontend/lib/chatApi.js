const DEFAULT_CHAT_API_PATH = "/api/chat";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const chatApiPath =
  process.env.NEXT_PUBLIC_CHAT_API_PATH || DEFAULT_CHAT_API_PATH;

const createChatUrl = () =>
  apiBaseUrl ? new URL(chatApiPath, apiBaseUrl).toString() : chatApiPath;

const normalizeIntent = (intent) =>
  typeof intent === "string" && intent.trim() ? intent : "일반";

const normalizeChatResponse = (data) => {
  if (!data || typeof data.reply !== "string") {
    throw new Error("Invalid chat response: reply is required.");
  }

  return {
    reply: data.reply,
    intent: normalizeIntent(data.intent),
    route: data.route,
    sources: Array.isArray(data.sources) ? data.sources : [],
    rag_domain: data.rag_domain,
    rag_domains: Array.isArray(data.rag_domains) ? data.rag_domains : [],
    rag_detail: data.rag_detail,
    rag_details: Array.isArray(data.rag_details) ? data.rag_details : [],
    answer_status: data.answer_status,
  };
};

export async function requestChatResponse({ message, language }) {
  const response = await fetch(createChatUrl(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      language,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat API request failed: ${response.status}`);
  }

  return normalizeChatResponse(await response.json());
}
