"use client";

import { Globe } from "lucide-react";

export default function Header() {
  const languages = [
    { code: "kr", label: "한국어" },
    { code: "en", label: "English" },
    { code: "zh", label: "中文" },
    { code: "ja", label: "日本語" },
  ];

  return (
    <header className="sticky top-0 z-50 flex h-14 items-center justify-between px-4 bg-[#003876] text-white border-b">

      {/* 왼쪽 제목 */}
      <div className="font-bold text-sm">
        KGU Smart Assistant
      </div>

      {/* 오른쪽 언어 버튼 */}
      <details className="relative">
        <summary className="flex cursor-pointer list-none items-center gap-1.5 rounded-lg border border-white/30 px-3 py-1.5 text-xs text-white transition-colors hover:bg-white/10 [&::-webkit-details-marker]:hidden">
          <Globe size={14} />
          <span className="font-medium">언어</span>
        </summary>

        {/* 드롭다운 */}
        <div className="absolute right-0 top-full z-50 mt-2 w-28 overflow-hidden rounded-xl border border-gray-100 bg-white py-1 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.1)]">
          <ul className="flex flex-col text-[13px] text-gray-700">
            {languages.map((lang) => (
              <li key={lang.code}>
                <a
                  href={`?lang=${lang.code}`}
                  className="block w-full px-4 py-2 text-left font-medium transition-colors hover:bg-gray-50 hover:text-[#003876]"
                >
                  {lang.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </details>

    </header>
  );
}
