const DEFAULT_BACKEND_API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_BACKEND_CHAT_API_PATH = "/api/v1/chat/chat";

const backendApiBaseUrl =
  process.env.BACKEND_API_BASE_URL || DEFAULT_BACKEND_API_BASE_URL;
const backendChatApiPath =
  process.env.BACKEND_CHAT_API_PATH || DEFAULT_BACKEND_CHAT_API_PATH;

const createBackendChatUrl = () =>
  new URL(backendChatApiPath, backendApiBaseUrl).toString();

const normalizeIntent = (intent) => {
  const allowedIntents = ["지도", "전화", "학식", "일반"];
  return allowedIntents.includes(intent) ? intent : "일반";
};

const normalizeChatData = (data) => {
  if (!data || typeof data.reply !== "string") {
    return null;
  }

  return {
    reply: data.reply,
    intent: normalizeIntent(data.intent),
  };
};

const hasBackendRuntimeError = (data) => {
  const reply = data?.reply ?? "";

  return [
    "API key not valid",
    "API_KEY_INVALID",
    "INVALID_ARGUMENT",
    "서버 오류가 발생했습니다",
  ].some((errorMessage) => reply.includes(errorMessage));
};

export async function POST(request) {
  try {
    const requestBody = await request.json();
    const backendResponse = await fetch(createBackendChatUrl(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(requestBody),
      cache: "no-store",
    });
    const contentType = backendResponse.headers.get("content-type") ?? "";
    const data = contentType.includes("application/json")
      ? await backendResponse.json()
      : null;
    const normalizedData = normalizeChatData(data);

    if (backendResponse.ok && normalizedData && !hasBackendRuntimeError(data)) {
      return Response.json(normalizedData);
    }

    return Response.json(
      {
        reply:
          "백엔드 응답을 읽지 못했습니다. 잠시 후 다시 시도해주세요.",
        intent: "일반",
      },
      { status: backendResponse.ok ? 502 : backendResponse.status },
    );
  } catch (error) {
    return Response.json(
      {
        reply:
          "백엔드 서버에 연결하지 못했습니다. 로컬 백엔드 실행 상태를 확인해주세요.",
        intent: "일반",
      },
      { status: 502 },
    );
  }
}
