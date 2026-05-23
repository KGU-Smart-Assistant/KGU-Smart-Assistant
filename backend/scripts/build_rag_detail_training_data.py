from __future__ import annotations

import json
from pathlib import Path

from build_rag_domain_training_data import DOMAINS


DETAIL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "period": (
        "{topic} 신청 기간 알려줘",
        "{topic} 마감일이 언제야?",
        "{topic} 일정 확인하고 싶어",
        "{topic} 접수는 언제부터야?",
    ),
    "eligibility": (
        "{topic} 지원 자격 알려줘",
        "{topic} 신청 조건이 뭐야?",
        "{topic} 대상자가 누구야?",
        "{topic} 받을 수 있는 기준 알려줘",
    ),
    "procedure": (
        "{topic} 신청 절차 알려줘",
        "{topic} 어떻게 신청해?",
        "{topic} 접수 방법 설명해줘",
        "{topic} 처리 절차가 궁금해",
    ),
    "required_documents": (
        "{topic} 제출서류 알려줘",
        "{topic} 신청서 양식 어디 있어?",
        "{topic} 서식 찾아줘",
        "{topic} 첨부파일 받을 수 있어?",
    ),
    "benefit": (
        "{topic} 지원 금액 알려줘",
        "{topic} 혜택이 뭐야?",
        "{topic} 감면 기준 알려줘",
        "{topic} 지원액은 얼마야?",
    ),
    "announcement_lookup": (
        "{topic} 관련 공지 찾아줘",
        "{topic} 안내문 어디서 확인해?",
        "{topic} 모집 공지 보여줘",
        "{topic} 결과 발표 공지 있어?",
    ),
    "summary": (
        "{topic} 내용 요약해줘",
        "{topic} 핵심만 정리해줘",
        "{topic} 중요한 내용만 알려줘",
        "{topic} 공지 요약 가능해?",
    ),
}

UNKNOWN_EXAMPLES: tuple[str, ...] = (
    "{topic}에 대해 궁금해",
    "{topic} 관련해서 알려줘",
    "{topic}은 어떻게 되는 거야?",
    "{topic}에 영향이 있어?",
)

COMPLEX_DETAIL_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("등록금 분납 신청 방법과 기간 알려줘", "procedure"),
    ("휴학 중 장학금 유지 여부랑 등록금 환불 기준 알려줘", "unknown"),
    ("교환학생 신청할 때 제출해야 하는 서류가 뭐야?", "required_documents"),
    ("복수전공을 하면 졸업학점이 달라지는지 궁금해", "unknown"),
    ("학기말 시험 기간이 학사일정에 나와 있어?", "period"),
    ("졸업요건 관련 제출서류가 있는지 알려줘", "required_documents"),
    ("학생증을 잃어버렸을 때 재발급 방법 알려줘", "procedure"),
    ("졸업 인증 기준을 확인하고 싶어", "eligibility"),
    ("다음 학기 학사 일정 전체를 보고 싶어", "announcement_lookup"),
    ("국가장학금 받을 수 있는 기준과 금액 알려줘", "eligibility"),
    ("등록금 감면 혜택이 있는지 궁금해", "benefit"),
    ("등록금 감면 받을 수 있는 혜택 알려줘", "benefit"),
    ("장학금 지원액과 혜택을 알고 싶어", "benefit"),
    ("해외파견 장학 혜택이 얼마나 되는지 알려줘", "benefit"),
    ("장학금 신청 방법이랑 기간 같이 알려줘", "procedure"),
    ("휴학 신청 절차와 마감일을 알려줘", "procedure"),
    ("등록금 환불 신청서와 제출 기간 알려줘", "required_documents"),
    ("교환학생 지원 조건이랑 장학 혜택 알려줘", "eligibility"),
    ("졸업요건 공지는 어디서 확인해?", "announcement_lookup"),
    ("전과 신청 관련 안내문 찾아줘", "announcement_lookup"),
    ("복학 신청서 양식 받을 수 있어?", "required_documents"),
    ("현장실습 모집공고 핵심만 정리해줘", "summary"),
    ("등록금 분납 혜택이나 감면이 있어?", "benefit"),
    ("근로장학 지원액이 얼마인지 알려줘", "benefit"),
    ("편입 모집요강 요약해줘", "summary"),
    ("수강취소가 졸업학점에 영향 있어?", "unknown"),
    ("다전공 신청 대상과 자격 알려줘", "eligibility"),
    ("학생증 재발급 신청서 서식 찾아줘", "required_documents"),
    ("학과 공지에 올라온 채용 결과 발표 확인해줘", "announcement_lookup"),
    ("학사일정에서 수강정정 마감일 알려줘", "period"),
    ("교직이수 신청 방법 알려줘", "procedure"),
    ("졸업인증 제출파일 어디에 있어?", "required_documents"),
    ("해외파견 프로그램 지원액과 혜택 알려줘", "benefit"),
    ("재입학하면 졸업요건이 바뀌어?", "unknown"),
    ("기숙사 신청 공지 요약 가능해?", "summary"),
    ("등록금 납부 안내문 어디서 봐?", "announcement_lookup"),
    ("전공변경 신청 자격이 되는지 궁금해", "eligibility"),
    ("수강신청 접수 절차 알려줘", "procedure"),
    ("복수전공 포기 기간은 언제야?", "period"),
    ("장학금 감면 기준 알려줘", "benefit"),
    ("편입 지원서 양식 찾아줘", "required_documents"),
    ("교환학생 선발 결과 발표 공지 있어?", "announcement_lookup"),
    ("졸업학점 기준과 전공학점 관계 알려줘", "unknown"),
    ("동아리 모집 안내 요약해줘", "summary"),
    ("자퇴 신청하면 등록금 환불되나요?", "unknown"),
    ("수업 신청 가능한 대상 알려줘", "eligibility"),
    ("교직과정 이수 신청 마감일 알려줘", "period"),
    ("현장실습 신청 절차랑 제출서류 알려줘", "required_documents"),
    ("학교 공지에서 장학금 안내 찾아줘", "announcement_lookup"),
    ("등록금 고지서 확인 방법 알려줘", "procedure"),
    ("학사일정 중요한 내용 정리해줘", "summary"),
    ("해외파견 지원 조건과 신청 방법 알려줘", "eligibility"),
    ("전과하면 복수전공도 다시 신청해야 해?", "unknown"),
    ("학생상담 신청 가능 시간 알려줘", "period"),
)

MULTI_DETAIL_EXAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("등록금 환불 신청서와 제출 기간 알려줘", ("required_documents", "period")),
    ("현장실습 신청 절차랑 제출서류 알려줘", ("required_documents", "procedure")),
    ("장학금 신청 방법이랑 기간 같이 알려줘", ("procedure", "period")),
    ("휴학 신청 절차와 마감일을 알려줘", ("procedure", "period")),
    ("등록금 분납 신청 방법과 기간 알려줘", ("procedure", "period")),
    ("교환학생 지원 조건이랑 장학 혜택 알려줘", ("eligibility", "benefit")),
    ("해외파견 지원 조건과 신청 방법 알려줘", ("eligibility", "procedure")),
    ("졸업요건 공지 핵심만 정리해줘", ("summary", "announcement_lookup")),
)


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    domain_train = backend_root / "app" / "data" / "rag_domain_train.jsonl"
    output_path = backend_root / "app" / "data" / "rag_detail_train.jsonl"

    rows: list[dict[str, object]] = []
    if domain_train.exists():
        rows.extend(_load_existing(domain_train))
    rows.extend(_generated_detail_rows())
    rows.extend(
        {"text": text, "route": "rag", "rag_detail": detail, "expected_details": [detail]}
        for text, detail in COMPLEX_DETAIL_EXAMPLES
    )
    rows.extend(
        {"text": text, "route": "rag", "rag_detail": details[0], "expected_details": list(details)}
        for text, details in MULTI_DETAIL_EXAMPLES
    )

    deduped = {str(row["text"]): row for row in rows}
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in deduped.values()) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(deduped)} examples to {output_path}")


def _load_existing(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        detail = row.get("rag_detail", "unknown")
        rows.append({"text": row["text"], "route": "rag", "rag_detail": detail, "expected_details": [detail]})
    return rows


def _generated_detail_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    topics = [topic for domain_topics in DOMAINS.values() for topic in domain_topics]
    for topic in topics:
        for detail, templates in DETAIL_TEMPLATES.items():
            for template in templates:
                rows.append({"text": template.format(topic=topic), "route": "rag", "rag_detail": detail, "expected_details": [detail]})
        for template in UNKNOWN_EXAMPLES:
            rows.append({"text": template.format(topic=topic), "route": "rag", "rag_detail": "unknown", "expected_details": ["unknown"]})
    return rows


if __name__ == "__main__":
    main()
