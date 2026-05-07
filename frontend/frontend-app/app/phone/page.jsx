"use client";

import { Suspense, useDeferredValue, useEffect, useMemo, useState } from "react";
import { Building2, PhoneCall, Search, X } from "lucide-react";
import { useSearchParams } from "next/navigation";

import { listContacts } from "@/lib/api";

const normalizeText = (value) =>
  String(value ?? "")
    .toLowerCase()
    .replace(/\s+/g, "");

function PhoneDirectoryContent() {
  const searchParams = useSearchParams();
  const requestedKeyword = searchParams.get("q") ?? "";
  const [keyword, setKeyword] = useState(requestedKeyword);
  const deferredKeyword = useDeferredValue(keyword);
  const [contacts, setContacts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setKeyword(requestedKeyword);
  }, [requestedKeyword]);

  useEffect(() => {
    let isMounted = true;

    async function loadContacts() {
      try {
        setIsLoading(true);
        setError("");
        const response = await listContacts();
        if (isMounted) {
          setContacts(response.contacts ?? []);
        }
      } catch (loadError) {
        if (isMounted) {
          setError(loadError.message);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    loadContacts();

    return () => {
      isMounted = false;
    };
  }, []);

  const filteredContacts = useMemo(() => {
    const normalizedKeyword = normalizeText(deferredKeyword);
    if (!normalizedKeyword) {
      return contacts;
    }

    return contacts.filter((contact) =>
      [
        contact.department_name,
        contact.phone_number,
        contact.description,
      ].some((value) => normalizeText(value).includes(normalizedKeyword)),
    );
  }, [contacts, deferredKeyword]);

  return (
    <main className="min-h-[calc(100dvh-136px)] bg-[#C6C9D4] px-4 py-5 text-[#27324b]">
      <section className="rounded-[28px] border border-[#d8dce6] bg-[#f9fbff] p-4 shadow-[0_18px_40px_rgba(42,53,80,0.08)]">
        <p className="text-xs font-extrabold uppercase tracking-[0.22em] text-[#F36F21]">
          KGU CONTACTS
        </p>
        <h1 className="mt-2 text-[26px] font-extrabold tracking-normal text-[#27324b]">
          전화번호 안내
        </h1>
        <p className="mt-2 break-keep text-sm leading-6 text-[#677489]">
          백엔드 연락처 API에서 불러온 부서 연락처입니다.
        </p>

        <form action="/phone" method="get" className="relative mt-5">
          <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#4f6486]" />
          <input
            name="q"
            type="search"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            placeholder="부서명, 전화번호, 설명 검색"
            className="h-12 w-full rounded-[20px] border border-[#d5dce7] bg-white pl-11 pr-11 text-sm text-[#27324b] outline-none transition placeholder:text-[#99a3b2] focus:border-[#003876] focus:ring-4 focus:ring-[#c7d8ee]"
          />
          {keyword ? (
            <a
              href="/phone"
              role="button"
              aria-label="검색어 지우기"
              className="absolute right-3 top-1/2 inline-flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-full border border-[#d7dfeb] bg-white text-[#607290] transition hover:border-[#95a8c7]"
            >
              <X className="h-4 w-4" />
            </a>
          ) : null}
        </form>
      </section>

      <section className="mt-5 space-y-3">
        {isLoading ? (
          <div className="rounded-[24px] border border-[#dce2eb] bg-white px-5 py-6 text-sm font-bold text-[#536076]">
            연락처를 불러오는 중입니다.
          </div>
        ) : null}

        {error ? (
          <div className="rounded-[24px] border border-rose-200 bg-rose-50 px-5 py-6 text-sm font-bold text-rose-600">
            연락처 API에 연결하지 못했습니다: {error}
          </div>
        ) : null}

        {!isLoading && !error && filteredContacts.length === 0 ? (
          <div className="rounded-[24px] border border-dashed border-[#d0d6e3] bg-white/90 px-6 py-10 text-center">
            <Search className="mx-auto h-5 w-5 text-[#003876]" />
            <p className="mt-3 text-sm font-bold text-[#536076]">
              검색 결과가 없습니다.
            </p>
          </div>
        ) : null}

        {filteredContacts.map((contact) => (
          <article
            key={contact.department_id}
            className="grid grid-cols-[44px_minmax(0,1fr)_auto] items-center gap-3 rounded-[24px] border border-[#dce2eb] bg-white px-4 py-4 shadow-[0_12px_28px_rgba(50,58,74,0.05)]"
          >
            <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[#C6C9D4] text-[#003876]">
              <Building2 className="h-5 w-5" />
            </div>

            <div className="min-w-0">
              <h2 className="truncate text-[16px] font-extrabold tracking-normal text-[#2a3550]">
                {contact.department_name}
              </h2>
              <p className="mt-1 text-sm font-bold text-[#536076]">
                {contact.phone_number}
              </p>
              {contact.description ? (
                <p className="mt-1 break-keep text-xs leading-5 text-[#6d7789]">
                  {contact.description}
                </p>
              ) : null}
            </div>

            <a
              href={contact.tel_uri}
              className="inline-flex min-h-10 shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-full border border-[#d7dfeb] bg-white px-4 py-2 text-xs font-extrabold text-[#003876] transition hover:border-[#95a8c7]"
              aria-label={`${contact.department_name} 전화 ${contact.phone_number}`}
            >
              <PhoneCall className="h-4 w-4" />
              전화
            </a>
          </article>
        ))}
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
