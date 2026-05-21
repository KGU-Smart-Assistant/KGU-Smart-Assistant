"use client";

import { Globe } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";

import { useLanguage } from "@/contexts/LanguageContext";

export default function Header() {
  const { currentLang, setCurrentLang } = useLanguage();
  const pathname = usePathname();
  const router = useRouter();

  const languages = [
    { code: "kr", label: "한국어" },
    { code: "en", label: "English" },
    { code: "zh", label: "中文" },
    { code: "ja", label: "日本語" },
  ];

  const selectedLanguage = languages.find((lang) => lang.code === currentLang);

  const selectLanguage = (code) => {
    const params = new URLSearchParams(window.location.search);
    params.set("lang", code);
    setCurrentLang(code);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  };

  return (
    <header className="sticky top-0 z-50 flex h-14 items-center justify-between border-b bg-[#003876] px-4 text-white">
      <div className="text-sm font-bold">KGU Smart Assistant</div>

      <details className="relative">
        <summary className="flex cursor-pointer list-none items-center gap-1.5 rounded-lg border border-white/30 px-3 py-1.5 text-xs text-white transition-colors hover:bg-white/10 [&::-webkit-details-marker]:hidden">
          <Globe size={14} />
          <span className="font-medium">
            {selectedLanguage?.label ?? "Language"}
          </span>
        </summary>

        <div className="absolute right-0 top-full z-50 mt-2 w-28 overflow-hidden rounded-xl border border-gray-100 bg-white py-1 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.1)]">
          <ul className="flex flex-col text-[13px] text-gray-700">
            {languages.map((lang) => (
              <li key={lang.code}>
                <button
                  type="button"
                  onClick={() => selectLanguage(lang.code)}
                  disabled={lang.code === currentLang}
                  className="block w-full px-4 py-2 text-left font-medium transition-colors hover:bg-gray-50 hover:text-[#003876] disabled:text-[#003876]"
                >
                  {lang.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </details>
    </header>
  );
}
