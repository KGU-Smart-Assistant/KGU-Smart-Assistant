# 의도 분류 서비스

from app.services.gemini_service import get_gemini_response

def classify_intent(user_input: str) -> str:
    """
    사용자 입력을 분석하여 의도를 분류한다.
    반환값: "지도" | "전화" | "일반"
    """

    # 전화·연락처를 먼저 (예: "도서관 전화번호"는 지도가 아니라 전화)
    if any(
        keyword in user_input
        for keyword in ["전화번호", "전화", "연락처", "문의", "번호"]
    ):
        return "전화"

    if any(
        keyword in user_input
        for keyword in [
            "어디",
            "위치",
            "찾아",
            "가는 길",
            "어딨",
            "어딨어",
            "위치 알려",
            "도서관",
            "학생회관",
            "강의동",
            "공학관",
            "정문",
            "후문",
            "복지관",
            "박물관",
        ]
    ):
        return "지도"

    prompt = f"""
    너는 학교 챗봇의 의도 분류기다.

    아래 문장을 반드시 하나의 카테고리로만 분류해라.

    카테고리:
    - 지도
    - 전화
    - 일반

    규칙:
    - 반드시 위 3개 중 하나만 선택
    - 다른 설명 절대 금지

    문장:
    {user_input}

    답변:
    """

    result = get_gemini_response(prompt)

    if not result:
        return "일반"

    result = result.strip()

    if result == "지도":
        return "지도"
    elif result == "전화":
        return "전화"
    else:
        return "일반"