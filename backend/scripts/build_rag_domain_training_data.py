from __future__ import annotations

import json
from pathlib import Path


DOMAINS: dict[str, tuple[str, ...]] = {
    "scholarship": ("장학금", "성적향상장학금", "국가장학금", "교내장학", "근로장학"),
    "course_registration": ("수강신청", "수강정정", "수강취소", "강의 신청", "수업 신청"),
    "academic_calendar": ("학사일정", "개강", "종강", "시험 기간", "학기 일정"),
    "academic_status": ("휴학", "복학", "자퇴", "재입학", "학적변동"),
    "major_change": ("전과", "전공변경", "소속변경", "전공 이동", "학과 변경"),
    "multi_major": ("다전공", "복수전공", "부전공", "연계전공", "융합전공"),
    "admission_transfer": ("편입", "재외국민 입학", "신입학", "모집요강", "입학전형"),
    "teaching_certification": ("교직이수", "교원자격", "교직과정", "교원자격증", "교직 신청"),
    "graduation": ("졸업요건", "졸업학점", "졸업인증", "전공학점", "필수이수"),
    "tuition": ("등록금", "분납", "환불", "납부", "고지서"),
    "document_materials": ("제출서류", "신청서", "양식", "서식", "자료실"),
    "student_life": ("학생증", "기숙사", "동아리", "상담", "학생생활"),
    "career_support": ("취업", "현장실습", "인턴", "채용", "진로"),
    "international_exchange": ("교환학생", "해외파견", "국제교류", "복수학위", "어학연수"),
    "department_notice": ("컴퓨터공학과 공지", "경영학과 공지", "학과 공지", "전공 공지", "학부 공지"),
    "general_notice": ("학교 공지", "전체 공지", "공지사항", "모집 안내", "결과 발표"),
}

DETAIL_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("period", "{topic} 신청 기간 알려줘"),
    ("period", "{topic} 마감일이 언제야?"),
    ("eligibility", "{topic} 지원 자격 알려줘"),
    ("eligibility", "{topic} 신청 조건이 어떻게 돼?"),
    ("procedure", "{topic} 신청 절차 설명해줘"),
    ("procedure", "{topic} 어떻게 신청해?"),
    ("required_documents", "{topic} 제출서류 알려줘"),
    ("required_documents", "{topic} 신청서 양식 어디 있어?"),
    ("announcement_lookup", "{topic} 관련 공지 찾아줘"),
    ("announcement_lookup", "{topic} 안내문 어디서 확인해?"),
)

MULTI_DOMAIN_EXAMPLES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("휴학하면 등록금이랑 장학금은 어떻게 돼?", ("academic_status", "tuition", "scholarship"), "procedure"),
    ("복학할 때 등록금 납부 기간도 같이 알려줘", ("academic_status", "tuition"), "period"),
    ("수강취소하면 장학금에 영향 있어?", ("course_registration", "scholarship"), "unknown"),
    ("졸업요건이랑 다전공 이수 조건 알려줘", ("graduation", "multi_major"), "eligibility"),
    ("전과하면 장학금 유지돼?", ("major_change", "scholarship"), "unknown"),
    ("기숙사 신청 기간이랑 제출서류 알려줘", ("student_life", "document_materials"), "period"),
    ("현장실습 신청 자격이랑 제출서류 알려줘", ("career_support", "document_materials"), "eligibility"),
    ("교환학생 모집 공지랑 제출서류 어디서 봐?", ("international_exchange", "document_materials"), "required_documents"),
    ("컴퓨터공학과 다전공 공지 알려줘", ("multi_major", "department_notice"), "announcement_lookup"),
    ("경영학과 장학 공지 확인하고 싶어", ("scholarship", "department_notice"), "announcement_lookup"),
    ("졸업인증 자료 제출 기간 알려줘", ("graduation", "document_materials"), "period"),
    ("등록금 환불 신청서 양식 어디 있어?", ("tuition", "document_materials"), "required_documents"),
    ("교직이수하면 졸업학점에 영향 있어?", ("teaching_certification", "graduation"), "unknown"),
    ("편입생도 장학금 받을 수 있어?", ("admission_transfer", "scholarship"), "eligibility"),
    ("재입학하면 등록금 환불 기준이 어떻게 돼?", ("academic_status", "tuition"), "unknown"),
    ("수강취소 신청서 서식 찾아줘", ("course_registration", "document_materials"), "required_documents"),
    ("복수전공 포기하면 졸업요건은 어떻게 돼?", ("multi_major", "graduation"), "unknown"),
    ("해외파견 장학금 신청 조건 알려줘", ("international_exchange", "scholarship"), "eligibility"),
    ("학생증 재발급 신청서 어디서 받아?", ("student_life", "document_materials"), "required_documents"),
    ("입학전형 결과 발표 공지 찾아줘", ("admission_transfer", "general_notice"), "announcement_lookup"),
    ("학사일정에서 수강정정 기간 확인하고 싶어", ("academic_calendar", "course_registration"), "period"),
    ("학교 전체 공지에서 교환학생 모집 찾아줘", ("general_notice", "international_exchange"), "announcement_lookup"),
    ("학과 공지에 올라온 현장실습 모집 알려줘", ("department_notice", "career_support"), "announcement_lookup"),
    ("등록금 고지서랑 분납 신청서 둘 다 필요해", ("tuition", "document_materials"), "required_documents"),
)


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    eval_path = backend_root / "app" / "data" / "rag_intent_eval.jsonl"
    failure_cases_path = backend_root / "app" / "data" / "intent_failure_cases.jsonl"
    output_path = backend_root / "app" / "data" / "rag_domain_train.jsonl"

    rows: list[dict[str, object]] = []
    rows.extend(_load_existing(eval_path))
    rows.extend(_load_failure_cases(failure_cases_path))
    rows.extend(_single_domain_rows())
    rows.extend(_multi_domain_rows())

    deduped = {str(row["text"]): row for row in rows}
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in deduped.values()) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(deduped)} examples to {output_path}")


def _load_existing(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["expected_domains"] = row.get("expected_domains") or [row["rag_domain"]]
        rows.append(row)
    return rows


def _load_failure_cases(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("expected_route") != "rag":
            continue
        expected_domains = row.get("expected_rag_domains") or [row["expected_rag_domain"]]
        rows.append(
            {
                "text": row["text"],
                "route": "rag",
                "rag_domain": row["expected_rag_domain"],
                "rag_detail": row.get("expected_rag_detail", "unknown"),
                "source_scope": "unknown",
                "expected_domains": expected_domains,
            }
        )
    return rows


def _single_domain_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain, topics in DOMAINS.items():
        for index, topic in enumerate(topics):
            for detail, template in DETAIL_TEMPLATES[index % 2 :: 2]:
                rows.append(
                    {
                        "text": template.format(topic=topic),
                        "route": "rag",
                        "rag_domain": domain,
                        "rag_detail": detail,
                        "source_scope": "department" if domain == "department_notice" else "unknown",
                        "expected_domains": [domain],
                    }
                )
    return rows


def _multi_domain_rows() -> list[dict[str, object]]:
    return [
        {
            "text": text,
            "route": "rag",
            "rag_domain": domains[0],
            "rag_detail": detail,
            "source_scope": "department" if "department_notice" in domains else "unknown",
            "expected_domains": list(domains),
        }
        for text, domains, detail in MULTI_DOMAIN_EXAMPLES
    ]


if __name__ == "__main__":
    main()
