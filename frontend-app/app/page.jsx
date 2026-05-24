"use client";

import { useState, useRef, useEffect } from "react";
import { getInitialMessages, getMockResponse } from "@/data/mockChatData";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import QuickActions from "@/components/QuickActions";
import { useLanguage } from "@/contexts/LanguageContext";

export default function Home() {
  const { currentLang } = useLanguage();
  // [프론트엔드 테스트용 State] 대화 목록을 관리합니다.
  const [messages, setMessages] = useState(() => getInitialMessages("kr"));
  const scrollRef = useRef(null);
  const nextMessageIdRef = useRef(1); // 초기 메시지가 1개이므로 1부터 시작

  useEffect(() => {
    // 언어 변경 시 채팅 기록이 초기 메시지뿐이라면, 해당 언어로 갱신
    if (messages.length <= 1) {
      setMessages(getInitialMessages(currentLang));
    }
  }, [currentLang]);

  // 렌더 중 Date.now를 호출하지 않도록, 이벤트마다 순차 ID를 발급합니다.
  const getNextMessageId = () => {
    nextMessageIdRef.current += 1;
    return nextMessageIdRef.current;
  };

  // 메시지가 추가될 때마다 자동으로 맨 아래로 스크롤합니다.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // 새로운 메시지를 전송하는 함수
  const handleSendMessage = (text, payload = null) => {
    // 1. 사용자 메시지 목록에 추가
    const newUserMessage = {
      id: getNextMessageId(),
      sender: "user",
      reply: text,
    };

    const loadingMessageId = getNextMessageId();
    const loadingMessage = {
      id: loadingMessageId,
      sender: "bot",
      isThinking: true,
      startTime: Date.now(),
    };

    setMessages((prev) => [...prev, newUserMessage, loadingMessage]);

    // 백엔드로 전송될 실제 데이터 (payload가 있으면 payload, 없으면 텍스트 원본)
    const dataToSend = payload || text;

    // 2. 가상의 봇 응답 스케줄링 (로딩 UI를 보기 위해 2.5초로 연장)
    setTimeout(() => {
      // API 요청 시 body에 들어갈 데이터 구조를 시뮬레이션
      const requestBody = {
        message: dataToSend,
        language: currentLang
      };
      const botResponse = getMockResponse(requestBody);
      const newBotMessage = {
        id: loadingMessageId,
        sender: "bot",
        reply: botResponse.reply,
        intent: botResponse.intent,
      };
      setMessages((prev) => prev.map(msg => msg.id === loadingMessageId ? newBotMessage : msg));
    }, 2500); //2500 이건 로딩 UI를 보기 위한 임시 수치임 백업 탑재시 0 혹은 삭제 바람
  };

  // h-[calc(100vh-136px)] matches viewport height minus Header(56px) and BottomNav(80px padding area approx)
  return (
    <div className="flex h-[calc(100vh-136px)] flex-col bg-[#C6C9D4]">

      {/* 채팅 메시지가 출력되는 스크롤 영역 */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-6 scrollbar-hide flex flex-col"
      >
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
      </div>

      {/* 하단 영역 (Quick Actions + Chat Input) */}
      <div className="relative w-full bg-[#003876] border-t border-[#003876] flex flex-col pt-2 shrink-0">
        {/* 자주 묻는 질문 (Quick Actions) 버튼 그룹 */}
        <QuickActions onActionClick={handleSendMessage} />

        {/* 텍스트 입력창 */}
        <ChatInput onSendMessage={handleSendMessage} />

        {/* 하단 padding-bottom에 의한 빈 공간을 덮어주기 위한 가상 요소 */}
        <div className="absolute top-full left-0 right-0 h-[100px] bg-[#003876]" />
      </div>
    </div>
  );
}
