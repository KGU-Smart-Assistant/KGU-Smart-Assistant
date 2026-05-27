import kr from "@/localisation_kr/mainpage.json";
import en from "@/localisation_en/mainpage.json";
import zh from "@/localisation_zh/mainpage.json";
import ja from "@/localisation_ja/mainpage.json";

const dictionaries = { kr, en, zh, ja };

// 초기 인사말 데이터 (현재 언어를 받아서 생성)
export const getInitialMessages = (lang = "kr") => [
  {
    id: 1,
    sender: "bot",
    reply: dictionaries[lang].mockBot.initialMessage,
    intent: "일반"
  }
];

// 사용자의 질문(또는 버튼 클릭)에 따라 적절한 목업 응답을 반환하는 함수
// 백엔드 요청을 시뮬레이션하기 위해 { message, language } 객체를 받습니다.
export const getMockResponse = (requestBody) => {
  const { message: query, language: lang = "kr" } = requestBody;
  const t = dictionaries[lang].mockBot;
  
  if (query.includes("학사일정") || query === "QA_1" || query.includes("Academic") || query.includes("校历") || query.includes("学年暦")) {
    return { reply: t.academicCalendar, intent: "일반" };
  } else if (query.includes("교환학생") || query === "QA_2" || query.includes("Exchange") || query.includes("交换生") || query.includes("交換留学生")) {
    return { reply: t.exchangeStudent, intent: "일반" };
  } else if (query.includes("등록금") || query.includes("장학") || query === "QA_3" || query.includes("Tuition") || query.includes("学费") || query.includes("学費")) {
    return { reply: t.tuition, intent: "일반" };
  } else if (query.includes("지도") || query.includes("위치") || query.includes("어디야") || query.includes("map") || query.includes("位置")) {
    return { reply: t.map, intent: "지도" };
  } else if (query.includes("전화번호") || query.includes("번호") || query.includes("phone") || query.includes("电话") || query.includes("電話")) {
    return { reply: t.phone, intent: "전화" };
  } else if (query.includes("학식") || query.includes("메뉴") || query.includes("cafeteria") || query.includes("食堂") || query.includes("メニュー")) {
    return { reply: t.cafeteria, intent: "학식" };
  } else {
    return { reply: t.fallback, intent: "일반" };
  }
};
