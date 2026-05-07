"use client";

import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import QuickActions from "@/components/QuickActions";
import { useLanguage } from "@/contexts/LanguageContext";
import { sendChatMessage } from "@/lib/api";

function getInitialMessages(t) {
  return [
    {
      id: 1,
      sender: "bot",
      reply: t("chat.initialMessage"),
      intent: "general",
    },
  ];
}

export default function Home() {
  const { currentLang, t } = useLanguage();
  // [프론트엔드 테스트용 State] 대화 목록을 관리합니다.
  const [messages, setMessages] = useState(() => getInitialMessages(t));
  const [isSending, setIsSending] = useState(false);
  const scrollRef = useRef(null);
  const nextMessageIdRef = useRef(1); // 초기 메시지가 1개이므로 1부터 시작

  useEffect(() => {
    // 언어 변경 시 채팅 기록이 초기 메시지뿐이라면, 해당 언어로 갱신
    if (messages.length <= 1) {
      setMessages(getInitialMessages(t));
    }
  }, [currentLang, messages.length, t]);

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
  const handleSendMessage = async (text) => {
    const trimmedText = text.trim();
    if (!trimmedText || isSending) {
      return;
    }

    // 1. 사용자 메시지 목록에 추가
    const newUserMessage = {
      id: getNextMessageId(),
      sender: "user",
      reply: trimmedText,
    };
    
    setMessages((prev) => [...prev, newUserMessage]);
    setIsSending(true);

    const dataToSend = trimmedText;

    // 2. 백엔드 API로 실제 봇 응답 요청
    setTimeout(async () => {
      try {
        const botResponse = await sendChatMessage(dataToSend);
        const newBotMessage = {
          id: getNextMessageId(),
          sender: "bot",
          reply: botResponse.reply,
          intent: botResponse.intent,
          route: botResponse.route,
          sources: botResponse.sources ?? [],
        };
        setMessages((prev) => [...prev, newBotMessage]);
      } catch (error) {
        const newBotMessage = {
          id: getNextMessageId(),
          sender: "bot",
          reply:
            "백엔드와 연결하지 못했습니다. API 서버가 실행 중인지 확인한 뒤 다시 시도해 주세요.",
          intent: "error",
          error: error.message,
        };
        setMessages((prev) => [...prev, newBotMessage]);
      } finally {
        setIsSending(false);
      }
    }, 600); // 0.6초 대기
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
        <ChatInput onSendMessage={handleSendMessage} disabled={isSending} />

        {/* 하단 padding-bottom에 의한 빈 공간을 덮어주기 위한 가상 요소 */}
        <div className="absolute top-full left-0 right-0 h-[100px] bg-[#003876]" />
      </div>
    </div>
  );
}
