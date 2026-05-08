from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


OUTPUT_PATH = Path("app/data/intent_training_seed.jsonl")


@dataclass(frozen=True)
class Example:
    text: str
    route: str
    db_intent: str = "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic seed data for intent training.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    examples = dedupe(build_examples())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            json.dumps(example.__dict__, ensure_ascii=False)
            for example in examples
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(examples)} examples to {output_path}")


def build_examples() -> list[Example]:
    return [
        *build_rag_examples(),
        *build_map_examples(),
        *build_phone_examples(),
        *build_relational_unknown_examples(),
        *build_weather_examples(),
        *build_llm_examples(),
        *build_edge_case_examples(),
    ]


def build_rag_examples() -> list[Example]:
    topics = [
        "성적향상 장학금",
        "국가장학금",
        "교내 장학금",
        "학업장려 장학금",
        "근로장학금",
        "성적우수 장학금",
        "복지장학금",
        "등록금 납부",
        "등록금 분할납부",
        "등록금 고지서 출력",
        "수강신청",
        "수강정정",
        "수강철회",
        "학사일정",
        "기말고사 일정",
        "중간고사 일정",
        "졸업요건",
        "졸업인증",
        "조기졸업",
        "전공 이수 학점",
        "교양 이수 학점",
        "휴학 신청",
        "복학 신청",
        "자퇴 신청",
        "재입학 신청",
        "비교과 프로그램",
        "청소년학과 전공이수자격원",
        "경영학과 공지",
        "컴퓨터공학전공 공지",
        "관광개발경영학과 공지",
        "자료실 첨부파일",
        "신청서 양식",
        "제출 서류 양식",
        "취업 프로그램",
        "현장실습",
        "인턴십",
        "기숙사 신청",
        "생활관 입사",
        "생활관 퇴사",
        "학생증 발급",
        "모바일 학생증",
        "교환학생 신청",
        "국제교류 프로그램",
        "어학성적 제출",
        "계절학기 신청",
        "학과 공지",
        "전체 공지사항",
        "학사 공지사항",
        "장학 공지사항",
    ]
    requests = [
        "어디에서 정보를 찾을 수 있어?",
        "공지 찾아줘",
        "공지 어디서 확인해?",
        "안내 어디서 확인해?",
        "정보 어디서 찾아?",
        "관련 내용 어디에 있어?",
        "신청 조건 알려줘",
        "신청 기간 알려줘",
        "제출 서류 알려줘",
        "안내문 내용 요약해줘",
        "학교 문서 기준으로 알려줘",
        "관련 공지 링크 찾아줘",
        "자격 기준 설명해줘",
        "최신 안내 확인해줘",
        "담당 부서 안내가 있는지 찾아줘",
        "지원 대상 알려줘",
        "신청 절차 정리해줘",
        "공지사항 기준으로 답해줘",
        "학교 홈페이지에서 확인할 수 있는 내용 알려줘",
    ]
    examples = [
        Example(f"{topic}은 {request}", "rag")
        for topic in topics
        for request in requests
    ]
    return examples[:180]


def build_map_examples() -> list[Example]:
    places = [
        "8강의동",
        "1강의동",
        "2강의동",
        "3강의동",
        "4강의동",
        "5강의동",
        "6강의동",
        "7강의동",
        "학생회관",
        "중앙도서관",
        "복지관",
        "공학관",
        "진리관",
        "덕문관",
        "호연관",
        "정문",
        "후문",
        "체육관",
        "학생식당",
        "기숙사",
        "전자정보관",
        "예술관",
        "교수회관",
        "산학협력관",
        "본관",
        "강의동",
        "운동장",
        "국제교육원",
        "박물관",
        "주차장",
        "버스정류장",
        "셔틀버스 승강장",
        "학생상담센터",
        "보건진료소",
        "입학처",
        "교무처",
        "학생지원팀",
        "장학지원팀",
        "취업진로처",
        "국제교류팀",
        "학사지원팀",
        "등록금 담당 부서",
    ]
    requests = [
        "어디야?",
        "위치 알려줘",
        "가는 길 알려줘",
        "지도에서 보여줘",
        "어디에 있어?",
        "찾아가는 방법 알려줘",
        "캠퍼스맵에서 찾아줘",
        "근처 건물 알려줘",
        "가려면 어디로 가야 해?",
        "캠퍼스 안에서 찾아줘",
        "현재 위치에서 가까워?",
    ]
    examples = [
        Example(f"{place}은 {request}", "relational_db", "map")
        for place in places
        for request in requests
    ]
    return examples[:180]


def build_phone_examples() -> list[Example]:
    offices = [
        "도서관",
        "학생지원팀",
        "청소년학과",
        "장학 담당 부서",
        "입학처",
        "교무처",
        "시설관리팀",
        "학생상담센터",
        "기숙사 사무실",
        "학과 사무실",
        "국제교류팀",
        "행정실",
        "취업지원팀",
        "총무팀",
        "수업 담당 부서",
        "등록금 담당 부서",
        "교환학생 담당자",
        "장애학생지원센터",
        "보건진료소",
        "예비군연대",
        "총학생회",
        "중앙동아리연합회",
        "생활관 행정실",
        "교직팀",
        "학사지원팀",
        "장학지원팀",
        "취업진로처",
        "국제교육원",
        "산학협력단",
        "정보전산원",
    ]
    requests = [
        "전화번호 알려줘",
        "연락처 알려줘",
        "문의 번호 알려줘",
        "대표 번호가 뭐야?",
        "전화 문의는 어느 번호로 하면 돼?",
        "사무실 번호 찾아줘",
        "통화 가능한 번호 알려줘",
        "담당 부서 번호 알려줘",
        "연결 가능한 전화번호 알려줘",
        "문의하려면 어디로 전화해?",
        "상담 번호 알려줘",
    ]
    examples = [
        Example(f"{office} {request}", "relational_db", "phone")
        for office in offices
        for request in requests
    ]
    return examples[:150]


def build_relational_unknown_examples() -> list[Example]:
    subjects = [
        "학교에 저장된 캠퍼스 기본 정보",
        "캠퍼스 데이터베이스의 기본 항목",
        "내부 DB에 있는 학교 데이터",
        "저장된 캠퍼스 장소 데이터",
        "학교 DB 기준 정보",
        "시스템에 등록된 학교 항목",
        "백엔드 DB에 저장된 캠퍼스 정보",
        "관리 중인 학교 기본 데이터",
        "저장된 학내 데이터",
        "서비스 DB의 캠퍼스 레코드",
        "등록된 학교 연락처와 장소 데이터",
        "캠퍼스 장소와 부서 정보",
        "시스템에 저장된 경기대 기본 데이터",
        "내부 데이터베이스의 캠퍼스 항목",
        "학교 챗봇 DB에 있는 정보",
    ]
    requests = [
        "조회해줘",
        "기준으로 답해줘",
        "관련 항목 있는지 확인해줘",
        "검색해줘",
        "목록을 확인해줘",
        "어떤 데이터가 있는지 알려줘",
        "DB 기준으로 분류해줘",
        "정확한 저장 데이터로 답해줘",
        "내부 데이터에서 확인 가능한지 봐줘",
    ]
    examples = [
        Example(f"{subject} {request}", "relational_db", "unknown")
        for subject in subjects
        for request in requests
    ]
    return examples[:90]


def build_weather_examples() -> list[Example]:
    times = [
        "오늘",
        "내일",
        "모레",
        "이번 주",
        "주말",
        "오후",
        "저녁",
        "아침",
        "지금",
        "다음 주 월요일",
        "등교 시간",
        "하교 시간",
        "축제 당일",
        "시험 기간",
    ]
    requests = [
        "수원 날씨 알려줘",
        "경기대 근처 비 와?",
        "기온 알려줘",
        "우산 필요해?",
        "강수확률 알려줘",
        "춥거나 더운지 알려줘",
        "날씨 예보 확인해줘",
        "비 올 가능성 있어?",
        "학교 갈 때 겉옷 필요해?",
        "캠퍼스 걸어다니기 괜찮아?",
        "야외 행사하기 괜찮을까?",
    ]
    examples = [
        Example(f"{time} {request}", "weather")
        for time in times
        for request in requests
    ]
    return examples[:100]


def build_edge_case_examples() -> list[Example]:
    return [
        Example("학생지원팀 어디야?", "relational_db", "map"),
        Example("장학지원팀 어디야?", "relational_db", "map"),
        Example("취업진로처 어디야?", "relational_db", "map"),
        Example("국제교류팀 어디야?", "relational_db", "map"),
        Example("학사지원팀 어디야?", "relational_db", "map"),
        Example("등록금 담당 부서 어디야?", "relational_db", "map"),
        Example("학생지원팀 위치 알려줘", "relational_db", "map"),
        Example("장학지원팀 위치 알려줘", "relational_db", "map"),
        Example("학생지원팀 찾아가는 길 알려줘", "relational_db", "map"),
        Example("장학지원팀 찾아가는 길 알려줘", "relational_db", "map"),
        Example("학생지원팀 전화번호 알려줘", "relational_db", "phone"),
        Example("장학지원팀 전화번호 알려줘", "relational_db", "phone"),
        Example("취업진로처 전화번호 알려줘", "relational_db", "phone"),
        Example("국제교류팀 전화번호 알려줘", "relational_db", "phone"),
        Example("학사지원팀 전화번호 알려줘", "relational_db", "phone"),
        Example("등록금 담당 부서 전화번호 알려줘", "relational_db", "phone"),
        Example("장학금 공지 어디서 확인해?", "rag"),
        Example("등록금 공지 어디서 확인해?", "rag"),
        Example("수강신청 공지 어디서 확인해?", "rag"),
        Example("졸업요건 안내 어디서 확인해?", "rag"),
    ]


def build_llm_examples() -> list[Example]:
    prompts = [
        "안녕",
        "고마워",
        "너는 누구야?",
        "오늘 기분 어때?",
        "간단하게 자기소개 해줘",
        "공부 집중하는 방법 알려줘",
        "대학교 생활 조언해줘",
        "파이썬이 뭐야?",
        "좋은 하루 보내",
        "질문 하나 해도 돼?",
        "시간 관리 방법 알려줘",
        "면접 준비 팁 알려줘",
        "리포트 잘 쓰는 법 알려줘",
        "팀플 갈등 해결 방법 알려줘",
        "영어 공부 방법 추천해줘",
        "간단한 농담 해줘",
        "오늘 할 일 정리하는 법 알려줘",
        "스트레스 줄이는 방법 알려줘",
        "발표 잘하는 팁 알려줘",
        "새로운 취미 추천해줘",
        "학교생활 적응 팁 알려줘",
        "새내기에게 조언해줘",
        "과제 미루지 않는 방법 알려줘",
        "팀플 잘하는 방법 알려줘",
        "시험 공부 계획 세워줘",
        "수업 필기 잘하는 법 알려줘",
        "동아리 고르는 팁 알려줘",
        "대학생 예산 관리 방법 알려줘",
        "공강 시간 활용법 알려줘",
        "학교 친구 사귀는 방법 알려줘",
    ]
    styles = [
        "",
        " 짧게",
        " 자세히",
        " 예시도 같이",
    ]
    examples = [
        Example(f"{prompt}{style}", "llm")
        for prompt in prompts
        for style in styles
    ]
    return examples[:100]


def dedupe(examples: list[Example]) -> list[Example]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Example] = []
    for example in examples:
        key = (example.text, example.route, example.db_intent)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


if __name__ == "__main__":
    main()
