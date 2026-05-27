const DEFAULT_CHAT_API_PATH = "/api/chat";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const chatApiPath =
  process.env.NEXT_PUBLIC_CHAT_API_PATH || DEFAULT_CHAT_API_PATH;

const createChatUrl = () =>
  apiBaseUrl ? new URL(chatApiPath, apiBaseUrl).toString() : chatApiPath;

const normalizeIntent = (intent) => {
  const allowedIntents = ["지도", "전화", "학식", "일반"];
  return allowedIntents.includes(intent) ? intent : "일반";
};

const normalizeChatResponse = (data) => {
  if (!data || typeof data.reply !== "string") {
    throw new Error("Invalid chat response: reply is required.");
  }

  return {
    reply: data.reply,
    intent: normalizeIntent(data.intent),
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
