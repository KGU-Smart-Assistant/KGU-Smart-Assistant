from datetime import datetime

from app.crawlers.document_quality import (
    build_attachment_fallback_text,
    filter_quality_documents,
    is_garbled_text,
    is_searchable_chunk_text,
    normalize_document_text,
    sanitize_title,
)
from app.schemas import Document


def _document(
    content: str,
    *,
    doc_id: str = "doc-1",
    source_type: str = "html",
    title: str = "Notice",
) -> Document:
    return Document(
        doc_id=doc_id,
        source_type=source_type,
        source_url=f"https://example.com/{doc_id}",
        title=title,
        content=content,
        category="notice",
        department="academic",
        published_at=None,
        collected_at=datetime(2026, 5, 1, 12, 0, 0),
    )


def test_filter_quality_documents_removes_short_html_documents() -> None:
    result = filter_quality_documents([_document("짧은 글")])

    assert result.documents == []
    assert result.removed_short == 1


def test_filter_quality_documents_removes_navigation_noise() -> None:
    content = "로그인 회원가입 개인정보처리방침 사이트맵 " + "메뉴 " * 40

    result = filter_quality_documents([_document(content)])

    assert result.documents == []
    assert result.removed_navigation_noise == 1


def test_filter_quality_documents_removes_html_listing_pages() -> None:
    content = "| 학사공지 | 2025.05.28 | 1180 | " * 10 + "비밀번호 입력 비밀번호 확인"
    document = _document(content)
    document.source_url = "https://ee.kgu.ac.kr/index.php?hCode=BOARD&page=list&pg=3"

    result = filter_quality_documents([document])

    assert result.documents == []
    assert result.removed_navigation_noise == 1


def test_filter_quality_documents_removes_career_list_pages() -> None:
    document = _document(
        "행사 수원캠퍼스 서울캠퍼스 캘린더 전체 접수대기중 접수중 운영중 종료 "
        "번호 프로그램 정보 신청기간 신청 정원 상태 " * 5,
        title="행사",
    )
    document.source_url = "https://job.kyonggi.ac.kr/ko/program/program3/suwon/list/all/3"

    result = filter_quality_documents([document])

    assert result.documents == []
    assert result.removed_navigation_noise == 1


def test_filter_quality_documents_keeps_career_detail_pages() -> None:
    document = _document(
        "행사 수원캠퍼스 서울캠퍼스 캘린더 2026년 1학기 상상사진관 무료 이력서 사진촬영 프로그램 "
        "신청대상 학부생 신청형태 개인 장소 복지관 문의 인재개발처 세부내용 " * 4,
        title="행사",
    )
    document.source_url = "https://job.kyonggi.ac.kr/ko/program/program3/suwon/view/666?p=1"

    result = filter_quality_documents([document])

    assert len(result.documents) == 1


def test_filter_quality_documents_removes_student_life_link_directory() -> None:
    document = _document(
        "대학소개 입학에서 취업까지 등록 휴학 복학 자원퇴학 전과 복수전공 장학안내 바로가기 " * 6,
        title="Ⅰ. 대학소개",
    )
    document.source_url = "https://www.kyonggi.ac.kr/www/contents.do?key=9346"

    result = filter_quality_documents([document])

    assert result.documents == []
    assert result.removed_navigation_noise == 1


def test_filter_quality_documents_keeps_informative_notice() -> None:
    content = (
        "2026학년도 성적우수장학금 신청 안내입니다. "
        "신청 기간은 5월 1일부터 5월 10일까지이며 제출 서류는 신청서와 성적증명서입니다. "
        "대상자는 재학생이며 자세한 내용은 학생지원처 공지를 확인해야 합니다."
    )

    result = filter_quality_documents([_document(content)])

    assert len(result.documents) == 1
    assert result.total_output == 1


def test_filter_quality_documents_replaces_attachment_with_link_fallback() -> None:
    result = filter_quality_documents(
        [
            _document(
                "정상적으로 추출된 첨부파일 본문이라도 검색 본문으로 쓰지 않습니다.",
                source_type="hwp",
                title="[국어국문] 졸업논문 안내",
            )
        ]
    )

    assert len(result.documents) == 1
    assert result.attachment_link_fallbacks == 1
    assert "원문 링크" in result.documents[0].content
    assert result.documents[0].source_url in result.documents[0].content


def test_filter_quality_documents_keeps_download_link_curriculum_page() -> None:
    document = _document(
        "인공지능전공 교육과정 2025학년도 교육과정 다운로드 미리보기",
        source_type="html",
        title="인공지능전공 교육과정",
    )
    document.attachment_urls = [
        "https://www.kyonggi.ac.kr/downloadContentsFile.do?fileNo=1500&key=9122"
    ]

    result = filter_quality_documents([document])

    assert len(result.documents) == 1
    assert result.attachment_link_fallbacks == 1
    assert "다운로드 URL" in result.documents[0].content
    assert document.attachment_urls[0] in result.documents[0].content


def test_normalize_document_text_cleans_whitespace() -> None:
    normalized = normalize_document_text(_document("장학금\n\n신청\t기간"))

    assert normalized == "장학금 신청 기간"


def test_filter_quality_documents_keeps_normalized_content() -> None:
    document = _document(
        "LMS 안내입니다. 온라인 강의와 원격수업을 확인할 수 있습니다. "
        "문의 전화와 이용 방법을 안내하는 본문입니다. "
        "학생은 매 학기 개설된 강좌를 확인하고 수업 자료와 공지를 조회할 수 있습니다.\n\n"
        "콘텐츠 정보 담당부서 원격교육지원센터\n맞춤설정 맞춤정보 어떤 정보를 찾고계시나요?",
        source_type="html",
        title="LMS",
    )
    document.source_url = "https://www.kyonggi.ac.kr/www/contents.do?key=7799"

    result = filter_quality_documents([document])

    assert len(result.documents) == 1
    assert "LMS 안내입니다." in result.documents[0].content
    assert "콘텐츠 정보" not in result.documents[0].content
    assert "맞춤설정" not in result.documents[0].content


def test_sanitize_title_removes_home_markdown_link() -> None:
    title = "* [HOME](https://www.kyonggi.ac.kr/www/index.do#default)"

    assert sanitize_title(title) == "제목 없음"


def test_sanitize_title_removes_broken_markdown_link_tail() -> None:
    title = "](https://ee.kgu.ac.kr/)"

    assert sanitize_title(title) == "제목 없음"


def test_is_searchable_chunk_text_rejects_low_value_and_garbled_text() -> None:
    assert not is_searchable_chunk_text("내역")
    assert not is_searchable_chunk_text("B 교직 복수전공 이수신청서")
    assert not is_searchable_chunk_text("\uf071\uf071\uf071\uf09e\uf09e\uf09e")
    assert is_searchable_chunk_text(
        "성적우수장학금은 직전 학기 성적과 이수학점을 기준으로 선발하며 자세한 기준은 공지사항을 확인합니다."
    )


def test_is_garbled_text_detects_private_use_characters() -> None:
    assert is_garbled_text("\uf071\uf071\uf071\uf09e\uf09e\uf09e")


def test_is_garbled_text_detects_bad_pdf_cjk_noise() -> None:
    text = (
        "KGI 2024 01 Q Today 4S HX H全 田 87 田 警巴星暑 77 B201 "
        "召今 对里 箱 研 请对 是 全号 基到 为 号合 草 召今 对里 箱 研 请对 是 全号 "
        "829000 15208133 11688133 21222 8241560 180"
    )

    assert is_garbled_text(text)


def test_is_garbled_text_detects_bad_numeric_table_noise() -> None:
    text = (
        "829,000 () *15.208.133 ）21（01-00169888-186-0 2 *11,688,133登71 "
        "221213 *11,611.221 09.17~12.16 *3,088 32212170 企：0 全号 对里 "
        "135-85-47313 186834564760 *8.251.50971 21222 166934564486 "
        "*3.355.889 7221222 *8.241.560初71 *180.08984"
    )

    assert is_garbled_text(text)


def test_build_attachment_fallback_text_includes_source_url() -> None:
    document = _document("bad", source_type="pdf", title="첨부 안내")

    fallback = build_attachment_fallback_text(document)

    assert "첨부파일 본문을 안정적으로 추출하지 못했습니다" in fallback
    assert document.source_url in fallback
