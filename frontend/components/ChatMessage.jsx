import Image from "next/image";
import { useState, useEffect } from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { ExternalLink } from "lucide-react";

// 개별 채팅 말풍선을 그려주는 공통 컴포넌트입니다.
export default function ChatMessage({ message, onLinkButtonClick }) {
  // 발신자가 챗봇인지 판단 (목업 데이터 기준)
  const isBot = message.sender === "bot";
  const { t } = useLanguage();
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    let interval;
    if (message.isThinking) {
      setElapsedSeconds(Math.floor((Date.now() - message.startTime) / 1000));
      interval = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - message.startTime) / 1000));
      }, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [message.isThinking, message.startTime]);

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
        {isBot && (
          <div className="flex items-center mb-1 ml-1">
            <span className="text-sm font-semibold text-gray-800 tracking-wide">기룡이</span>
            {message.isThinking && (
              <span className="text-gray-500 font-normal ml-2 text-xs">
                ({t("mockBot.thinking").replace("{{time}}", elapsedSeconds)})
              </span>
            )}
          </div>
        )}

        {/* 말풍선 박스. 챗봇은 흰색 배경, 사용자는 경기대 컬러(#003876) 사용 */}
        <div
          className={`max-w-[80vw] px-4 py-2 rounded-2xl break-words overflow-wrap-anywhere ${isBot
            ? "bg-white text-gray-900 rounded-tl-none border border-gray-200 shadow-sm"
            : "bg-[#003876] text-white rounded-tr-none"
            }`}
        >
          {message.isThinking ? (
            <div className="flex space-x-1 items-center h-5 px-1 py-1">
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
          ) : (
            <>
              {/* 실제 텍스트 대답 내용 */}
              <p className="text-sm leading-relaxed whitespace-pre-wrap break-words overflow-wrap-anywhere" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{message.reply}</p>

              {/* 링크 버튼들 표시 (하위 항목들 - 클릭 시 챗봇으로 전송) */}
              {message.links && message.links.length > 0 && (
                <div className="mt-3 flex flex-col gap-2">
                  {message.links.map((link, index) => (
                    <button
                      key={index}
                      type="button"
                      onClick={() => onLinkButtonClick && onLinkButtonClick(t(link.label), link.url)}
                      className="flex items-center justify-center gap-2 rounded-xl bg-[#003876] px-4 py-2.5 text-sm font-bold text-white transition hover:bg-[#00264d] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#003876] focus-visible:ring-offset-2"
                    >
                      <span className="break-words text-center">{t(link.label)}</span>
                    </button>
                  ))}
                </div>
              )}

              {/* 단일 링크 바로가기 (실제 링크로 이동) */}
              {message.finalLink && (
                <a
                  href={message.finalLink.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-3 flex items-center justify-center gap-2 rounded-xl bg-[#003876] px-4 py-2.5 text-sm font-bold text-white transition hover:bg-[#00264d] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#003876] focus-visible:ring-offset-2"
                >
                  <span className="break-words">{t(message.finalLink.label)}{t("infoLinks.learnMore")}</span>
                  <ExternalLink className="h-4 w-4 shrink-0" />
                </a>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}