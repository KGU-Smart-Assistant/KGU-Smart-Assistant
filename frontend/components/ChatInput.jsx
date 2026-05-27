"use client";

import { useState } from "react";
import { Send } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

// 사용자 입력을 받아 채팅을 전송하는 컴포넌트
export default function ChatInput({ onSendMessage }) {
  const { t } = useLanguage();
  // [프론트엔드 상호작용 테스트용 State]
  const [text, setText] = useState("");

  const handleSend = () => {
    if (text.trim() === "") return;
    onSendMessage(text); // 부모 컴포넌트에 사용자가 입력한 메시지 전달
    setText(""); // 입력창 초기화
  };

  const handleKeyDown = (e) => {
    // 엔터키를 눌렀을 때 메시지 전송 (한글 입력 시 조합 문제로 인해 isComposing 체크를 하기도 하지만 여기선 기본 엔터만 처리)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    // 네비게이션 앱 하단 등에 안 붙도록 배경을 주고 패딩 설정
    <div className="p-4 bg-[#003876] flex items-center">
      
      {/* 텍스트 입력창. focus시 테두리에 경기대 메인컬러 노출 */}
      <input
        type="text"
        placeholder={t("chatInput.placeholder")}
        className="flex-1 bg-white text-gray-900 placeholder-gray-500 px-4 py-3 rounded-full text-sm outline-none focus:ring-2 focus:ring-white transition-all shadow-inner"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      
      {/* 전송 버튼. 경기대 메인컬러 배경에 lucide-react의 Send 아이콘 적용 */}
      <button 
        onClick={handleSend}
        className="ml-2 w-10 h-10 flex-shrink-0 rounded-full bg-[#003876] border border-white flex items-center justify-center focus:outline-none hover:bg-blue-800 transition-colors"
      >
        <Send size={18} color="white" />
      </button>
    </div>
  );
}
