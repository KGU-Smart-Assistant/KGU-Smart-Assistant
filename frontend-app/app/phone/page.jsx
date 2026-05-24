"use client";

import {
  Suspense,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Building2, MapPin, PhoneCall, Search, X } from "lucide-react";
import { useSearchParams } from "next/navigation";

import { useLanguage } from "@/contexts/LanguageContext";
import { campusTabs, phoneDirectory } from "@/data/phoneDirectory";

const ALL_CATEGORY = "all";
const SUPPORTED_LANGUAGES = ["kr", "en", "zh", "ja"];

const CATEGORY_KEY_BY_NAME = {
  대표번호: "mainLine",
  행정: "administration",
  학사: "academics",
  입학: "admissions",
  학생지원: "studentSupport",
  시설관리: "facilities",
};

const BUTTON_BASE_CLASS =
  "rounded-full border font-extrabold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#003876] focus-visible:ring-offset-2";
const BUTTON_ACTIVE_CLASS =
  "border-[#003876] bg-[#003876] text-white shadow-[0_12px_24px_rgba(0,56,118,0.18)]";
const BUTTON_IDLE_CLASS =
  "border-[#d7dfeb] bg-white text-[#003876] hover:border-[#95a8c7]";
const PHONE_BUTTON_CLASS =
  "inline-flex min-h-10 shrink-0 items-center justify-center gap-2 whitespace-nowrap px-4 py-2 text-xs";
const ICON_BUTTON_CLASS =
  "inline-flex h-8 w-8 items-center justify-center rounded-full border border-[#d7dfeb] bg-white text-[#607290] transition hover:border-[#95a8c7] active:border-[#003876] active:bg-[#003876] active:text-white";

const categoryToneMap = {
  mainLine: {
    badge: "border-[#F36F21] bg-[#F9C1B0] text-[#9b4a1f]",
    icon: "bg-[#F9C1B0] text-[#F36F21]",
  },
  administration: {
    badge: "border-[#003876] bg-[#C6C9D4] text-[#003876]",
    icon: "bg-[#C6C9D4] text-[#003876]",
  },
  academics: {
    badge: "border-[#ffb400] bg-[#FFDBAE] text-[#8a6400]",
    icon: "bg-[#FFDBAE] text-[#b57d00]",
  },
  admissions: {
    badge: "border-[#F36F21] bg-[#F9C1B0] text-[#9b4a1f]",
    icon: "bg-[#F9C1B0] text-[#F36F21]",
  },
  studentSupport: {
    badge: "border-[#00AB39] bg-[#B7DCBC] text-[#0c6d2d]",
    icon: "bg-[#B7DCBC] text-[#00AB39]",
  },
  facilities: {
    badge: "border-[#003876] bg-[#C6C9D4] text-[#003876]",
    icon: "bg-[#C6C9D4] text-[#003876]",
  },
};

const defaultTone = {
  badge: "border-[#003876] bg-[#C6C9D4] text-[#003876]",
  icon: "bg-[#C6C9D4] text-[#003876]",
};

const normalizeText = (value) =>
  String(value ?? "")
    .toLowerCase()
    .replace(/\s+/g, "");

const createTelHref = (phone) => `tel:${String(phone).replace(/[^\d+]/g, "")}`;

const getCategoryKey = (category) => CATEGORY_KEY_BY_NAME[category] ?? ALL_CATEGORY;

const getTranslatedText = (t, key, language, fallback = "") => {
  const keyPath = `phone.${key}`;
  const translated = t(keyPath, language);
  return translated === keyPath ? fallback : translated;
};

const matchesKeyword = (keyword, values) =>
  values.some((value) => normalizeText(value).includes(keyword));

function PhoneDirectoryContent() {
  const { currentLang, setCurrentLang, t } = useLanguage();
  const searchParams = useSearchParams();
  const requestedLanguage = searchParams.get("lang");
  const requestedCampus = searchParams.get("campus");
  const requestedCategory = searchParams.get("category");
  const requestedKeyword = searchParams.get("q") ?? "";
  const activeLanguage = SUPPORTED_LANGUAGES.includes(requestedLanguage)
    ? requestedLanguage
    : currentLang;
  const activeCampus = campusTabs.some((item) => item.campus === requestedCampus)
    ? requestedCampus
    : campusTabs[0]?.campus ?? "";
  const [keyword, setKeyword] = useState(requestedKeyword);
  const deferredKeyword = useDeferredValue(keyword);

  const phoneT = useCallback(
    (key, fallback = "") => getTranslatedText(t, key, activeLanguage, fallback),
    [activeLanguage, t],
  );
  const currentCampusInfo =
    campusTabs.find((item) => item.campus === activeCampus) ?? campusTabs[0];
  const currentCampusKey = currentCampusInfo?.id ?? "suwon";

  useEffect(() => {
    if (
      requestedLanguage &&
      SUPPORTED_LANGUAGES.includes(requestedLanguage) &&
      requestedLanguage !== currentLang
    ) {
      setCurrentLang(requestedLanguage);
    }
  }, [currentLang, requestedLanguage, setCurrentLang]);

  useEffect(() => {
    setKeyword(requestedKeyword);
  }, [requestedKeyword]);

  const campusSections = useMemo(
    () => phoneDirectory.filter((section) => section.campus === activeCampus),
    [activeCampus],
  );

  const activeCategory =
    requestedCategory &&
    campusSections.some((section) => section.category === requestedCategory)
      ? requestedCategory
      : ALL_CATEGORY;

  const campusCopy = {
    label: phoneT(`campuses.${currentCampusKey}.label`, currentCampusInfo?.label),
    address: phoneT(`campuses.${currentCampusKey}.address`, currentCampusInfo?.address),
    description: phoneT(
      `campuses.${currentCampusKey}.description`,
      currentCampusInfo?.description,
    ),
  };

  const createPhoneHref = ({
    campus = activeCampus,
    category = activeCategory,
    q = keyword,
  } = {}) => {
    const params = new URLSearchParams();

    if (campus) {
      params.set("campus", campus);
    }

    params.set("lang", activeLanguage);

    if (category && category !== ALL_CATEGORY) {
      params.set("category", category);
    }

    if (q.trim()) {
      params.set("q", q.trim());
    }

    return `/phone?${params.toString()}`;
  };

  const availableCategories = useMemo(
    () => [
      { id: ALL_CATEGORY, label: phoneT("categories.all") },
      ...campusSections.map((section) => {
        const categoryKey = getCategoryKey(section.category);

        return {
          id: section.category,
          label: phoneT(`categories.${categoryKey}`, section.category),
        };
      }),
    ],
    [campusSections, phoneT],
  );

  const filteredSections = useMemo(() => {
    const normalizedKeyword = normalizeText(deferredKeyword);

    return campusSections
      .filter(
        (section) =>
          activeCategory === ALL_CATEGORY || section.category === activeCategory,
      )
      .map((section) => {
        const categoryKey = getCategoryKey(section.category);
        const localizedCategory = phoneT(`categories.${categoryKey}`, section.category);
        const sectionMatched =
          normalizedKeyword.length > 0 &&
          matchesKeyword(normalizedKeyword, [
            section.category,
            localizedCategory,
            currentCampusInfo?.campus,
            campusCopy.label,
            campusCopy.address,
            campusCopy.description,
          ]);

        const departments = section.departments
          .map((department) => ({
            ...department,
            localizedName: phoneT(
              `departments.${department.id}.name`,
              department.name,
            ),
            localizedDescription: phoneT(
              `departments.${department.id}.description`,
              department.description,
            ),
          }))
          .filter((department) => {
            if (!normalizedKeyword || sectionMatched) {
              return true;
            }

            return matchesKeyword(normalizedKeyword, [
              department.name,
              department.phone,
              department.description,
              department.localizedName,
              department.localizedDescription,
              section.category,
              localizedCategory,
            ]);
          });

        return {
          ...section,
          categoryKey,
          localizedCategory,
          departments,
        };
      })
      .filter((section) => section.departments.length > 0);
  }, [
    activeCategory,
    campusCopy.address,
    campusCopy.description,
    campusCopy.label,
    campusSections,
    currentCampusInfo?.campus,
    deferredKeyword,
    phoneT,
  ]);

  const campusContactCount = campusSections.reduce(
    (total, section) => total + section.departments.length,
    0,
  );

  const visibleCount = filteredSections.reduce(
    (total, section) => total + section.departments.length,
    0,
  );

  const formatDepartmentCount = (count) =>
    `${count}${phoneT("units.departments", " departments")}`;

  return (
    <main className="min-h-[calc(100dvh-136px)] bg-[#C6C9D4] px-4 py-5 text-[#27324b]">
      <section className="rounded-[28px] border border-[#d8dce6] bg-[#f9fbff] p-4 shadow-[0_18px_40px_rgba(42,53,80,0.08)]">
        <div className="min-w-0">
          <p className="text-xs font-extrabold uppercase tracking-[0.22em] text-[#F36F21]">
            {phoneT("page.eyebrow")}
          </p>
          <h1 className="mt-2 text-[26px] font-extrabold tracking-normal text-[#27324b]">
            {phoneT("page.title")}
          </h1>
          <p className="mt-2 break-keep text-sm leading-6 text-[#677489]">
            {phoneT("page.description")}
          </p>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-2 rounded-full bg-[#aeb9ce] p-1.5">
          {campusTabs.map((tab) => {
            const isActive = tab.campus === activeCampus;

            return (
              <a
                key={tab.id}
                href={createPhoneHref({
                  campus: tab.campus,
                  category: ALL_CATEGORY,
                })}
                role="button"
                aria-pressed={isActive}
                className={`${BUTTON_BASE_CLASS} px-4 py-3 text-sm ${
                  isActive ? BUTTON_ACTIVE_CLASS : BUTTON_IDLE_CLASS
                }`}
              >
                {phoneT(`campuses.${tab.id}.label`, tab.label)}
              </a>
            );
          })}
        </div>

        {currentCampusInfo ? (
          <div className="mt-4 rounded-[22px] border border-[#c2cedf] bg-[#eef3fb] p-4">
            <div className="flex items-start gap-2 text-sm text-[#526076]">
              <MapPin className="mt-0.5 h-4 w-4 shrink-0 text-[#F36F21]" />
              <p className="font-semibold leading-5">{campusCopy.address}</p>
            </div>

            <p className="mt-3 break-keep text-xs leading-5 text-[#8c96a7]">
              {campusCopy.description}
            </p>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <div className="rounded-[18px] border border-[#ffb400]/35 bg-[#FFDBAE] px-3 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-[#8a6400]">
                  {phoneT("page.categoryStat")}
                </p>
                <p className="mt-1 text-lg font-extrabold text-[#6f5200]">
                  {campusSections.length}
                </p>
              </div>
              <div className="rounded-[18px] border border-[#00AB39]/30 bg-[#B7DCBC] px-3 py-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-[#0c6d2d]">
                  {phoneT("page.contactsStat")}
                </p>
                <p className="mt-1 text-lg font-extrabold text-[#0c6d2d]">
                  {visibleCount}
                  <span className="ml-1 text-xs font-bold text-[#476954]">
                    / {campusContactCount}
                  </span>
                </p>
              </div>
            </div>
          </div>
        ) : null}

        <form action="/phone" method="get" className="relative mt-5">
          <input type="hidden" name="campus" value={activeCampus} />
          <input type="hidden" name="lang" value={activeLanguage} />
          {activeCategory !== ALL_CATEGORY ? (
            <input type="hidden" name="category" value={activeCategory} />
          ) : null}
          <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#4f6486]" />
          <input
            name="q"
            type="search"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            placeholder={phoneT("page.searchPlaceholder")}
            className="h-12 w-full rounded-[20px] border border-[#d5dce7] bg-white pl-11 pr-11 text-sm text-[#27324b] outline-none transition placeholder:text-[#99a3b2] focus:border-[#003876] focus:ring-4 focus:ring-[#c7d8ee]"
          />
          {keyword ? (
            <a
              href={createPhoneHref({ q: "" })}
              role="button"
              aria-label={phoneT("page.clearSearchAria")}
              className={`absolute right-3 top-1/2 -translate-y-1/2 ${ICON_BUTTON_CLASS}`}
            >
              <X className="h-4 w-4" />
            </a>
          ) : null}
        </form>

        <div className="mt-4 flex gap-2 overflow-x-auto pb-1 [scrollbar-width:none]">
          {availableCategories.map((category) => {
            const isActive = category.id === activeCategory;

            return (
              <a
                key={category.id}
                href={createPhoneHref({ category: category.id })}
                role="button"
                aria-pressed={isActive}
                className={`${BUTTON_BASE_CLASS} shrink-0 px-4 py-2 text-xs ${
                  isActive ? BUTTON_ACTIVE_CLASS : BUTTON_IDLE_CLASS
                }`}
              >
                {category.label}
              </a>
            );
          })}
        </div>
      </section>

      <section className="mt-5 space-y-5">
        {filteredSections.length > 0 ? (
          filteredSections.map((section) => {
            const tone = categoryToneMap[section.categoryKey] ?? defaultTone;

            return (
              <section
                key={`${section.campus}-${section.category}`}
                className="space-y-3 rounded-[28px] border border-[#b5c2d8] bg-[#b5c0d4] p-4"
              >
                <div className="flex items-center justify-between gap-3">
                  <div
                    className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-xs font-extrabold ${tone.badge}`}
                  >
                    <Building2 className="h-4 w-4" />
                    {section.localizedCategory}
                  </div>
                  <p className="text-xs font-bold text-[#7e8997]">
                    {formatDepartmentCount(section.departments.length)}
                  </p>
                </div>

                <div className="space-y-3">
                  {section.departments.map((department) => (
                    <article
                      key={department.id}
                      className="grid grid-cols-[44px_minmax(0,1fr)_auto] items-center gap-3 rounded-[24px] border border-[#dce2eb] bg-white px-4 py-4 shadow-[0_12px_28px_rgba(50,58,74,0.05)]"
                    >
                      <div
                        className={`flex h-11 w-11 items-center justify-center rounded-full ${tone.icon}`}
                      >
                        <Building2 className="h-5 w-5" />
                      </div>

                      <div className="min-w-0">
                        <h2 className="truncate text-[16px] font-extrabold tracking-[-0.02em] text-[#2a3550]">
                          {department.localizedName}
                        </h2>
                        <p className="mt-1 text-sm font-bold text-[#536076]">
                          {department.phone}
                        </p>
                        <p className="mt-1 break-keep text-xs leading-5 text-[#6d7789]">
                          {department.localizedDescription}
                        </p>
                      </div>

                      <a
                        href={createTelHref(department.phone)}
                        className={`${BUTTON_BASE_CLASS} ${PHONE_BUTTON_CLASS} ${BUTTON_IDLE_CLASS} visited:text-[#003876] active:border-[#003876] active:bg-[#003876] active:text-white`}
                        aria-label={`${department.localizedName} ${phoneT(
                          "page.callAction",
                        )} ${department.phone}`}
                      >
                        <PhoneCall className="h-4 w-4" />
                        {phoneT("page.callAction")}
                      </a>
                    </article>
                  ))}
                </div>
              </section>
            );
          })
        ) : (
          <div className="rounded-[28px] border border-dashed border-[#d0d6e3] bg-white/90 px-6 py-10 text-center shadow-[0_12px_28px_rgba(50,58,74,0.05)]">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-[#C6C9D4] text-[#003876]">
              <Search className="h-5 w-5" />
            </div>
            <h2 className="mt-4 text-lg font-extrabold text-[#27324b]">
              {phoneT("page.noResultsTitle")}
            </h2>
            <p className="mt-2 break-keep text-sm leading-6 text-[#6d7789]">
              {phoneT("page.noResultsDescription")}
            </p>
          </div>
        )}
      </section>
    </main>
  );
}

export default function PhonePage() {
  return (
    <Suspense fallback={null}>
      <PhoneDirectoryContent />
    </Suspense>
  );
}
