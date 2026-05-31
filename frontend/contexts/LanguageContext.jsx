"use client";

import { createContext, useContext, useState } from "react";

import krMain from "@/localisation_kr/mainpage.json";
import enMain from "@/localisation_en/mainpage.json";
import zhMain from "@/localisation_zh/mainpage.json";
import jaMain from "@/localisation_ja/mainpage.json";

import krMap from "@/localisation_kr/mappage.json";
import enMap from "@/localisation_en/mappage.json";
import zhMap from "@/localisation_zh/mappage.json";
import jaMap from "@/localisation_ja/mappage.json";

import krPhone from "@/localisation_kr/phone.json";
import enPhone from "@/localisation_en/phone.json";
import zhPhone from "@/localisation_zh/phone.json";
import jaPhone from "@/localisation_ja/phone.json";

const dictionaries = {
  kr: { ...krMain, ...krMap, ...krPhone },
  en: { ...enMain, ...enMap, ...enPhone },
  zh: { ...zhMain, ...zhMap, ...zhPhone },
  ja: { ...jaMain, ...jaMap, ...jaPhone },
};

const LanguageContext = createContext();
const defaultLanguage = "kr";

const getInitialLanguage = () => {
  if (typeof window === "undefined") {
    return defaultLanguage;
  }

  const lang = new URLSearchParams(window.location.search).get("lang");
  return dictionaries[lang] ? lang : defaultLanguage;
};

export function LanguageProvider({ children }) {
  const [currentLang, setCurrentLang] = useState(getInitialLanguage);

  const t = (keyPath, language = currentLang) => {
    const keys = keyPath.split(".");
    let current = dictionaries[language] ?? dictionaries[currentLang];
    for (const key of keys) {
      if (current && current[key] !== undefined) {
        current = current[key];
      } else {
        return keyPath;
      }
    }
    return current;
  };

  return (
    <LanguageContext.Provider value={{ currentLang, setCurrentLang, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export const useLanguage = () => useContext(LanguageContext);
