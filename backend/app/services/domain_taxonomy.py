from __future__ import annotations

from dataclasses import dataclass

CANONICAL_DOMAIN_LABELS: tuple[str, ...] = (
    "scholarship",
    "course_registration",
    "academic_calendar",
    "academic_status",
    "major_change",
    "multi_major",
    "admission_transfer",
    "teaching_certification",
    "graduation",
    "tuition",
    "document_materials",
    "student_life",
    "career_support",
    "international_exchange",
    "department_notice",
    "general_notice",
    "faq",
    "unknown",
)

DOMAINS = set(CANONICAL_DOMAIN_LABELS)

CANONICAL_DETAIL_LABELS: tuple[str, ...] = (
    "period",
    "eligibility",
    "procedure",
    "required_documents",
    "benefit",
    "announcement_lookup",
    "summary",
    "unknown",
)

DETAILS = set(CANONICAL_DETAIL_LABELS)

LEGACY_DOMAIN_ALIASES = {
    "academic": "academic_calendar",
    "academic_schedule": "academic_calendar",
    "schedule": "academic_calendar",
    "leave_of_absence": "academic_status",
    "return_to_school": "academic_status",
    "double_major": "multi_major",
    "minor": "multi_major",
    "transfer": "admission_transfer",
    "admission": "admission_transfer",
    "teaching": "teaching_certification",
    "exchange": "international_exchange",
    "international": "international_exchange",
    "support": "scholarship",
    "scholarship_support": "scholarship",
    "materials": "document_materials",
    "career": "career_support",
    "notice": "general_notice",
    "university_notices": "general_notice",
    "department_sources": "department_notice",
    "graduation_requirements": "graduation",
    "unknown": "unknown",
}

DOMAIN_FILTERS: dict[str, list[str]] = {
    "scholarship": ["scholarship", "general_notice", "department_notice"],
    "tuition": ["tuition", "general_notice", "department_notice"],
    "course_registration": ["course_registration", "academic_calendar", "general_notice", "department_notice"],
    "academic_calendar": ["academic_calendar", "general_notice", "department_notice"],
    "academic_status": ["academic_status", "academic_calendar", "general_notice", "department_notice"],
    "major_change": ["major_change", "academic_status", "graduation", "general_notice", "department_notice"],
    "multi_major": ["multi_major", "academic_status", "graduation", "general_notice", "department_notice"],
    "admission_transfer": ["admission_transfer", "general_notice", "department_notice"],
    "teaching_certification": ["teaching_certification", "graduation", "general_notice", "department_notice"],
    "graduation": ["graduation", "general_notice", "department_notice"],
    "document_materials": ["document_materials", "academic_calendar", "general_notice", "department_notice"],
    "student_life": ["student_life", "general_notice", "department_notice"],
    "career_support": ["career_support", "general_notice", "department_notice"],
    "international_exchange": ["international_exchange", "general_notice", "department_notice"],
    "department_notice": ["department_notice"],
    "general_notice": ["general_notice"],
    "faq": ["faq"],
}

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "scholarship": ("장학", "장학금", "성적향상장학금", "국가장학금", "교내장학", "학자금", "수혜", "감면"),
    "tuition": ("등록금 납부", "등록금 분납", "등록금 환불", "고지서", "납부", "분납", "환불"),
    "course_registration": ("수강신청", "수강 정정", "수강정정", "수강취소", "수기수강", "수강 철회"),
    "academic_calendar": ("학사일정", "개강", "종강", "중간고사", "기말고사", "시험기간", "시험 기간", "휴학", "복학", "성적공시", "강의평가"),
    "academic_status": ("학적", "휴학", "복학", "자퇴", "제적", "재입학", "학사경고"),
    "major_change": ("전과", "전공변경", "전공 변경", "소속변경", "소속 변경"),
    "multi_major": ("다전공", "복수전공", "부전공", "연계전공", "융합전공"),
    "admission_transfer": ("입학", "편입", "편입학", "모집요강", "전형", "입시"),
    "teaching_certification": ("교직", "교원자격", "교직이수", "교육실습"),
    "graduation": ("졸업", "졸업요건", "졸업 학점", "전공 학점", "교양 학점", "졸업인증", "학위"),
    "document_materials": ("자료실", "자료", "첨부파일", "파일", "양식", "서식", "신청서", "제출서류", "pdf", "hwp", "docx"),
    "student_life": ("학생생활", "학생증", "동아리", "상담", "통학", "셔틀", "기숙사", "생활관", "복지", "식당", "편의시설"),
    "career_support": ("취업", "진로", "커리어", "현장실습", "인턴", "채용", "비교과", "취업지원"),
    "international_exchange": ("국제교류", "교환학생", "해외파견", "어학연수", "유학생", "글로벌"),
    "department_notice": ("학과 공지", "전공 공지", "단과대", "학과별", "청소년학과", "컴퓨터공학과", "전공 안내"),
    "general_notice": ("학교 공지", "전체 공지", "공지사항", "공지", "모집", "선발", "결과 발표", "일반 안내"),
    "faq": ("faq", "자주 묻는 질문", "질문", "답변 모음"),
}

DOMAIN_PRIORITY = {
    "scholarship": 8,
    "tuition": 8,
    "course_registration": 8,
    "academic_status": 8,
    "major_change": 8,
    "multi_major": 8,
    "admission_transfer": 8,
    "teaching_certification": 8,
    "graduation": 8,
    "document_materials": 7,
    "academic_calendar": 6,
    "career_support": 6,
    "international_exchange": 6,
    "student_life": 5,
    "department_notice": 4,
    "general_notice": 1,
    "faq": 1,
}

DETAIL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "period": ("기간", "일정", "언제", "마감", "날짜", "기한", "시기"),
    "eligibility": ("대상", "자격", "조건", "가능한가", "해당", "선발 기준"),
    "procedure": ("신청 방법", "방법", "절차", "어떻게", "접수", "처리"),
    "required_documents": ("제출서류", "제출 서류", "신청서", "양식", "증빙", "서식", "첨부파일"),
    "benefit": ("금액", "혜택", "지원액", "감면", "지원 내용", "수혜"),
    "announcement_lookup": ("공지", "안내", "모집", "확인", "찾아", "어디서", "발표"),
    "summary": ("요약", "정리", "핵심", "간단히"),
}


@dataclass(frozen=True)
class DomainClassification:
    domain: str
    detail: str


def normalize_domain(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().casefold()
    return LEGACY_DOMAIN_ALIASES.get(normalized, normalized if normalized in DOMAINS else None)


def normalize_detail(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in DETAILS else None


def infer_domain(text: str, *, fallback: str = "unknown") -> str:
    normalized = text.casefold()
    best: tuple[str, int, int] | None = None
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(keyword.casefold() in normalized for keyword in keywords)
        if count <= 0:
            continue
        priority = DOMAIN_PRIORITY.get(domain, 0)
        if best is None or (priority, count) > (best[1], best[2]):
            best = (domain, priority, count)
    return best[0] if best else fallback


def infer_detail(text: str, *, fallback: str = "unknown") -> str:
    normalized = text.casefold()
    for detail, keywords in DETAIL_KEYWORDS.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            return detail
    return fallback


def classify_domain(
    *,
    title: str = "",
    content: str = "",
    source_url: str = "",
    source_name: str = "",
    source_domain: str | None = None,
    legacy_category: str | None = None,
) -> DomainClassification:
    normalized_source = normalize_domain(source_domain) or normalize_domain(legacy_category)
    text = " ".join(part for part in (title, content[:1200], source_url, source_name) if part)
    inferred = infer_domain(text, fallback="unknown")

    # Structured crawler sources are already scoped to a domain. Do not let
    # incidental menu/footer words in the page body move them into another domain.
    if normalized_source in {
        "academic_calendar",
        "academic_status",
        "major_change",
        "multi_major",
        "admission_transfer",
        "teaching_certification",
        "document_materials",
        "faq",
        "student_life",
        "career_support",
        "scholarship",
        "graduation",
        "tuition",
        "course_registration",
        "international_exchange",
    }:
        domain = normalized_source
    elif inferred != "unknown":
        domain = inferred
    elif normalized_source:
        domain = normalized_source
    else:
        domain = "general_notice"
    return DomainClassification(domain=domain, detail=infer_detail(text))
