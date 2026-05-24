"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { getInitialMessages, getMockResponse } from "@/data/mockChatData";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import InfoLinkDropdown from "@/components/InfoLinkDropdown";
import QuickActions from "@/components/QuickActions";
import { useLanguage } from "@/contexts/LanguageContext";
import { requestChatResponse } from "@/lib/chatApi";

export default function Home() {
  const { currentLang, t } = useLanguage();
  // [프론트엔드 테스트용 State] 대화 목록을 관리합니다.
  const [messages, setMessages] = useState([]);
  const scrollRef = useRef(null);
  const nextMessageIdRef = useRef(1); // 초기 메시지가 1개이므로 1부터 시작

  const displayedMessages = useMemo(
    () => (messages.length > 0 ? messages : getInitialMessages(currentLang)),
    [currentLang, messages],
  );

  // 렌더 중 Date.now를 호출하지 않도록, 이벤트마다 순차 ID를 발급합니다.
  const getNextMessageId = () => {
    nextMessageIdRef.current += 1;
    return nextMessageIdRef.current;
  };

  // 새채팅 시작 함수
  const handleNewChat = () => {
    setMessages([]);
    nextMessageIdRef.current = 1;
  };

  // 메시지가 추가될 때마다 자동으로 맨 아래로 스크롤합니다.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [displayedMessages]);

  // 새로운 메시지를 전송하는 함수
  const handleSendMessage = async (text, payload = null) => {
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
    setMessages((prev) => [
      ...(prev.length > 0 ? prev : getInitialMessages(currentLang)),
      newUserMessage,
      loadingMessage,
    ]);
    // 백엔드로 전송될 실제 데이터 (payload가 있으면 payload, 없으면 텍스트 원본)
    const dataToSend = payload || text;
    
    // 2. API 호출 및 봇 응답 처리
    const requestBody = {
      message: dataToSend,
      language: currentLang,
    };

    try {
      const botResponse = await requestChatResponse(requestBody);
      const newBotMessage = {
        id: loadingMessageId, // 로딩 메시지 ID 재사용
        sender: "bot",
        reply: botResponse.reply,
        intent: botResponse.intent,
      };
      // 로딩 메시지를 실제 메시지로 교체
      setMessages((prev) => prev.map(msg => 
        msg.id === loadingMessageId ? newBotMessage : msg
      ));
    } catch (error) {
      console.warn("Falling back to mock chat response.", error);
      const botResponse = getMockResponse(requestBody);
      const newBotMessage = {
        id: loadingMessageId,
        sender: "bot",
        reply: botResponse.reply,
        intent: botResponse.intent,
      };
      setMessages((prev) => prev.map(msg => 
        msg.id === loadingMessageId ? newBotMessage : msg
      ));
    }
  };

  // 링크 버튼 클릭 시 챗봇 메시지 생성 (finalLink 제공)
  const handleLinkButtonClick = async (linkLabel, linkUrl) => {
    const userMessageText = `${linkLabel}${t("infoLinks.curious")}`;
    const newUserMessage = {
      id: getNextMessageId(),
      sender: "user",
      reply: userMessageText,
    };

    const botMessage = {
      id: getNextMessageId(),
      sender: "bot",
      reply: `${linkLabel}${t("infoLinks.moreInfo")}`,
      intent: "일반",
      finalLink: { label: linkLabel, url: linkUrl },
    };

    setMessages((prev) => [
      ...(prev.length > 0 ? prev : getInitialMessages(currentLang)),
      newUserMessage,
      botMessage,
    ]);
  };

  // 링크 클릭 시 챗봇 메시지로 처리
  const handleLinkClick = async (title, links, isGroup) => {
    // 1. 사용자 메시지 생성
    const translatedTitle = t(title); // 번역 키를 실제 텍스트로 변환
    const userMessageText = `${translatedTitle}${t("infoLinks.curious")}`;
    
    const newUserMessage = {
      id: getNextMessageId(),
      sender: "user",
      reply: userMessageText,
    };

    // 2. 봇 응답 메시지 (즉시)
    let botMessage;
    
    if (isGroup) {
      // 그룹 클릭: 여러 버튼 표시
      botMessage = {
        id: getNextMessageId(),
        sender: "bot",
        reply: `${translatedTitle}${t("infoLinks.moreInfo")}\n\n${t("infoLinks.whichInfo")}`,
        intent: "일반",
        links: links, // 여러 링크 배열
      };
    } else {
      // 개별 링크 클릭 (드롭다운): 바로 finalLink 제공
      botMessage = {
        id: getNextMessageId(),
        sender: "bot",
        reply: `${translatedTitle}${t("infoLinks.moreInfo")}`,
        intent: "일반",
        finalLink: links[0], // 단일 링크
      };
    }

    setMessages((prev) => [
      ...(prev.length > 0 ? prev : getInitialMessages(currentLang)),
      newUserMessage,
      botMessage,
    ]);
  };

  // h-[calc(100vh-136px)] matches viewport height minus Header(56px) and BottomNav(80px padding area approx)
  return (
    <div className="flex h-[calc(100vh-136px)] flex-col bg-[#C6C9D4]">

      {/* 새채팅 버튼 */}
      <div className="flex justify-end px-4 pt-3 pb-2">
        <button
          onClick={handleNewChat}
          className="flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-sm font-bold text-[#003876] shadow-md transition hover:bg-[#f0f4f8] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#003876]"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14"/>
          </svg>
          <span>{t("newChat")}</span>
        </button>
      </div>

      {/* 채팅 메시지가 출력되는 스크롤 영역 */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-6 scrollbar-hide flex flex-col"
      >
        {displayedMessages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} onLinkButtonClick={handleLinkButtonClick} />
        ))}
      </div>

      {/* 하단 영역 (Quick Actions + Chat Input) */}
      <div className="relative w-full bg-[#003876] border-t border-[#003876] flex flex-col pt-2 shrink-0">
        {/* 자주 묻는 질문 (Quick Actions) 버튼 그룹 */}
        <QuickActions onActionClick={handleSendMessage} />

        <InfoLinkDropdown onLinkClick={handleLinkClick} />

        {/* 텍스트 입력창 */}
        <ChatInput onSendMessage={handleSendMessage} />

        {/* 하단 padding-bottom에 의한 빈 공간을 덮어주기 위한 가상 요소 */}
        <div className="absolute top-full left-0 right-0 h-[100px] bg-[#003876]" />
      </div>
    </div>
  );
}