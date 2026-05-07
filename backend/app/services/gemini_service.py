from datetime import datetime
from zoneinfo import ZoneInfo

from google import genai

from app.core.config import settings
from app.schemas import SearchResult


_APP_TIMEZONE = ZoneInfo("Asia/Seoul")


def _current_context() -> str:
    now = datetime.now(_APP_TIMEZONE)
    return (
        "Runtime context:\n"
        f"- Current date: {now.date().isoformat()}\n"
        f"- Current time: {now.strftime('%H:%M:%S %Z')}\n"
        "- Timezone: Asia/Seoul\n"
        "- Treat dates before the current date as past, the current date as present, "
        "and dates after the current date as future.\n"
        "- Do not use the model training cutoff as today's date.\n"
        "- For weather forecasts or current weather, do not invent live weather data. "
        "If no weather data source is provided, explain that real-time weather data is required.\n"
    )


def _with_current_context(prompt: str) -> str:
    return f"{_current_context()}\nUser/task prompt:\n{prompt}"


def _call_gemini(prompt: str) -> str:
    try:
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=_with_current_context(prompt),
        )

        if response and response.text:
            return response.text.strip()
        return ""

    except Exception as exc:
        error_msg = str(exc)

        print("\n" + "=" * 50)
        print(f"GEMINI ERROR: {error_msg}")
        print("=" * 50 + "\n")

        if "429" in error_msg:
            return (
                "현재 Gemini API 사용량 제한에 도달했습니다. "
                "잠시 후 다시 시도해 주세요."
            )
        if "503" in error_msg or "UNAVAILABLE" in error_msg:
            return (
                "현재 Gemini 모델 수요가 높아 응답을 생성하지 못했습니다. "
                "잠시 후 다시 시도해 주세요."
            )
        if "404" in error_msg:
            return (
                "설정된 Gemini 모델을 찾을 수 없습니다. "
                "GEMINI_MODEL 환경변수를 확인해 주세요."
            )

        return f"서버 오류가 발생했습니다: {error_msg}"


def get_gemini_response(user_input: str) -> str:
    return _call_gemini(user_input)


def get_gemini_response_with_context(
    user_input: str,
    context: str | list[SearchResult],
) -> str:
    if isinstance(context, list):
        return _get_gemini_response_with_search_results(user_input, context)
    return _get_gemini_response_with_text_context(user_input, context)


def _get_gemini_response_with_text_context(user_input: str, context: str) -> str:
    prompt = f"""
You are a Korean university assistant for Kyonggi University.
Answer the user's question using only the information in the context.
If the context does not contain enough information, say that the available
information is insufficient and ask for a more specific question.
Do not invent dates, eligibility rules, amounts, office names, or URLs.

Context:
{context}

User question:
{user_input}

Answer in Korean:
"""
    return _call_gemini(prompt)


def _get_gemini_response_with_search_results(
    user_input: str,
    search_results: list[SearchResult],
) -> str:
    context = _format_search_context(search_results)
    prompt = f"""
아래 경기대학교 수집 자료만 근거로 답변하세요.
자료에 없는 내용은 모른다고 말하세요.
신청 기간, 자격, 제출 서류, 금액, 제한 조건처럼 자료에 직접 없는 항목은 추측하지 마세요.
자료가 부족하면 확인 가능한 공지 제목과 URL을 안내하세요.

참고 자료:
{context}

질문:
{user_input}

답변:
"""
    answer = _call_gemini(prompt)
    if not answer:
        return "관련 자료를 바탕으로 답변을 생성하지 못했습니다."
    return f"{answer}\n\n{_format_sources(search_results)}"


def _format_search_context(search_results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(search_results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[자료 {index}]",
                    f"제목: {result.title}",
                    f"URL: {result.source_url}",
                    f"내용: {result.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_sources(search_results: list[SearchResult]) -> str:
    seen_urls: set[str] = set()
    lines = ["출처:"]
    for result in search_results:
        if result.source_url in seen_urls:
            continue
        seen_urls.add(result.source_url)
        lines.append(f"- {result.title}: {result.source_url}")
    return "\n".join(lines)
