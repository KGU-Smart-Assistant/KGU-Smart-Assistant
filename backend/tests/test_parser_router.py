from types import SimpleNamespace

from app.crawlers.parsing.parser_router import ParserRouter
from app.crawlers.parsing.schemas import ParseContext


def test_parser_router_uses_faq_parser_for_faq_category() -> None:
    router = ParserRouter()
    result = SimpleNamespace(
        markdown=SimpleNamespace(fit_markdown="FAQ\n총게시물 : 0\n질문 답변 검색"),
        metadata={"title": "FAQ - 예시학과"},
    )

    parsed = router.parse(
        result=result,
        context=ParseContext(
            url="https://example.com/selectBbsNttList.do?bbsNo=950",
            category="faq",
            department="example",
        ),
    )

    assert parsed is None


def test_parser_router_skips_korean_empty_faq_list_page() -> None:
    router = ParserRouter()
    result = SimpleNamespace(
        markdown=SimpleNamespace(
            fit_markdown=(
                "FAQ\n게시물 검색\n총게시물 : _0_ 건 페이지 : _1_ / 1\n"
                "검색결과가 없습니다."
            )
        ),
        metadata={"title": "FAQ - 호텔경영전공"},
    )

    parsed = router.parse(
        result=result,
        context=ParseContext(
            url="https://example.com/selectBbsNttList.do?bbsNo=978",
            category="faq",
            department="hotel_management",
        ),
    )

    assert parsed is None


def test_parser_router_keeps_non_empty_faq_list_page() -> None:
    router = ParserRouter()
    result = SimpleNamespace(
        markdown=SimpleNamespace(
            fit_markdown="FAQ\n질문 답변 검색\n휴학은 어디서 신청하나요?\n학과 사무실 방문"
        ),
        metadata={"title": "FAQ - 예시학과"},
    )

    parsed = router.parse(
        result=result,
        context=ParseContext(
            url="https://example.com/selectBbsNttList.do?bbsNo=950",
            category="faq",
            department="example",
        ),
    )

    assert parsed is not None
    assert parsed.title == "FAQ - 예시학과"
    assert parsed.content == "휴학은 어디서 신청하나요?\n학과 사무실 방문"


def test_parser_router_uses_schedule_parser_for_schedule_category() -> None:
    router = ParserRouter()
    result = SimpleNamespace(
        markdown=SimpleNamespace(fit_markdown="2026년 4월 학사일정"),
        metadata={"title": "학사일정 - 예시학과"},
    )

    parsed = router.parse(
        result=result,
        context=ParseContext(
            url="https://example.com/selectTnSchafsSchdulListUS.do?key=1",
            category="academic_schedule",
            department="example",
        ),
    )

    assert parsed is not None
    assert parsed.title == "학사일정 - 예시학과"


def test_parser_router_skips_global_navigation_title_candidates() -> None:
    router = ParserRouter()
    result = SimpleNamespace(
        markdown=SimpleNamespace(
            fit_markdown=(
                "* [경기대학교](https://www.kyonggi.ac.kr/www/index.do)\n"
                "* [입시홈페이지](http://enter.kyonggi.ac.kr/index.do)\n"
                "주메뉴 열기 입체조형학과\n"
                "* 인쇄\n"
                "입체조형학과 2022학년도 1학기 실험실습비 사용내역\n"
                "_작성자_ 입체조형학과\n"
            )
        ),
        metadata={"title": "* 경기대학교"},
        links={"internal": []},
    )

    parsed = router.parse(
        result=result,
        context=ParseContext(
            url="https://www.kyonggi.ac.kr/3Dimensional_Art/selectBbsNttView.do",
            category="materials",
            department="dimensional_art",
        ),
    )

    assert parsed is not None
    assert parsed.title == "입체조형학과 2022학년도 1학기 실험실습비 사용내역"


def test_parser_router_applies_site_specific_parser_options() -> None:
    router = ParserRouter()
    result = SimpleNamespace(
        markdown=SimpleNamespace(
            fit_markdown="* 경기대학교\n2025학년도 협력병원 건강검진 안내드립니다."
        ),
        metadata={"title": "* 경기대학교"},
        links={"internal": []},
    )

    parsed = router.parse(
        result=result,
        context=ParseContext(
            url="https://www.kyonggi.ac.kr/u_tour/selectBbsNttView.do?bbsNo=684&nttNo=1",
            category="materials",
            department="tourism_culture_college",
        ),
    )

    assert parsed is not None
    assert parsed.title == "2025학년도 협력병원 건강검진 안내드립니다."
