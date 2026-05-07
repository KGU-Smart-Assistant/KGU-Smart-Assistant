import Image from "next/image";

// 개별 채팅 말풍선을 그려주는 공통 컴포넌트입니다.
export default function ChatMessage({ message }) {
  // 발신자가 챗봇인지 판단 (목업 데이터 기준)
  const isBot = message.sender === "bot";

  return (
    <div className={`flex w-full mb-4 ${isBot ? "justify-start" : "justify-end"}`}>
      {/* 챗봇일 경우에만 프로필 마스코트 이미지 표시 */}
      {isBot && (
        <div className="flex-shrink-0 mr-3 mt-1">
          {/* 마스코트 이미지 컨테이너 (정원형 깎임 처리) */}
          <div className="w-12 h-12 rounded-full flex items-center justify-center overflow-hidden bg-white border border-gray-700">
            <Image src="/mascot.png" alt="기룡이" width={48} height={48} className="object-cover" />
          </div>
        </div>
      )}

      {/* 말풍선 본문 영역 */}
      <div className={`flex flex-col ${isBot ? "items-start" : "items-end"}`}>
        {isBot && <span className="text-sm font-semibold text-gray-800 mb-1 ml-1 tracking-wide">기룡이</span>}

        {/* 말풍선 박스. 챗봇은 흰색 배경, 사용자는 경기대 컬러(#003876) 사용 */}
        <div
          className={`max-w-[80vw] px-4 py-2 rounded-2xl ${isBot
              ? "bg-white text-gray-900 rounded-tl-none border border-gray-200 shadow-sm"
              : "bg-[#003876] text-white rounded-tr-none"
            }`}
        >
          {/* 실제 텍스트 대답 내용 */}
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.reply}</p>

          {/* [목업/테스트용 코드 영역] */}
          {/* 백엔드 연동 시 전달될 'intent' (의도) 값에 따라 시각적으로 다른 컴포넌트가 떠야 함을 보여주기 위한 가이드/목업 요소입니다. */}
          {/* 추후 진석/재승님이 만드실 실제 '지도'나 '전화' 컴포넌트로 이를 교체해야 합니다. */}

          {message.intent === "지도" && (
            <div className="mt-3 bg-gray-50 p-3 rounded-lg flex items-center text-gray-600 text-xs border border-gray-200">
              📍 [지도 컴포넌트 영역 - 목업 UI]
            </div>
          )}
          {message.intent === "전화" && (
            <div className="mt-3 bg-gray-50 p-3 rounded-lg flex items-center text-gray-600 text-xs border border-gray-200">
              📞 [전화 컴포넌트 영역 - 목업 UI]
            </div>
          )}
          {message.intent === "학식" && (
            <div className="mt-3 bg-gray-50 p-3 rounded-lg flex items-center text-gray-600 text-xs border border-gray-200">
              🍱 [학식 컴포넌트 영역 - 목업 UI]
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
