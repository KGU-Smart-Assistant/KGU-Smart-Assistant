"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  Building2,
  Compass,
  Info,
  Locate,
  MapPin,
  MapPinned,
  Route,
  Search,
  X,
} from "lucide-react";

import { buildingData, campusMapData } from "@/data/mapData";
import { useLanguage } from "@/contexts/LanguageContext";

const KAKAO_MAP_SCRIPT_ID = "kakao-map-sdk";
const CARD_CLASS =
  "min-w-0 overflow-hidden rounded-[28px] border border-[#d4deee] bg-white/94 shadow-[0_18px_45px_rgba(0,56,118,0.08)] backdrop-blur";
const ACTION_BUTTON_CLASS =
  "inline-flex items-center justify-center gap-2 rounded-full border border-[#c7d4e7] bg-white px-4 py-3 text-sm font-semibold text-[#003876] transition hover:-translate-y-0.5 hover:bg-[#eef3fb] hover:shadow-[0_12px_24px_rgba(0,56,118,0.1)]";

let kakaoMapLoaderPromise;

function createRawBounds(buildings, padding) {
  const latitudes = buildings.map((building) => building.lat);
  const longitudes = buildings.map((building) => building.lng);

  return {
    south: Math.min(...latitudes) - padding.lat,
    north: Math.max(...latitudes) + padding.lat,
    west: Math.min(...longitudes) - padding.lng,
    east: Math.max(...longitudes) + padding.lng,
  };
}

const CAMPUS_BOUNDS = createRawBounds(buildingData, campusMapData.boundsPadding);
const ALL_BUILDING_TAG = "map.filter.all";
const BUILDING_TAG_OPTIONS = [
  ALL_BUILDING_TAG,
  ...new Set(buildingData.map((building) => building.tag)),
];
const MOBILE_PANEL_TABS = [
  { id: "selected", icon: Building2 },
  { id: "list", icon: MapPinned },
  { id: "source", icon: Info },
];

function matchesSearch(building, term, t) {
  if (!term) return true;

  const text = [
    t(building.name),
    t(building.shortName),
    t(building.description),
    t(building.summary),
    t(building.tag),
    t(building.searchKeyword),
  ]
    .join(" ")
    .toLowerCase();

  return text.includes(term);
}

function matchesTag(building, activeTag) {
  return activeTag === ALL_BUILDING_TAG || building.tag === activeTag;
}

function clampCoordinate(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatCoordinate(value) {
  return value.toFixed(6);
}

function isInsideCampus(lat, lng) {
  return (
    lat >= CAMPUS_BOUNDS.south &&
    lat <= CAMPUS_BOUNDS.north &&
    lng >= CAMPUS_BOUNDS.west &&
    lng <= CAMPUS_BOUNDS.east
  );
}

function createLatLng(kakao, lat, lng) {
  return new kakao.maps.LatLng(lat, lng);
}

function createCampusBounds(kakao) {
  return new kakao.maps.LatLngBounds(
    createLatLng(kakao, CAMPUS_BOUNDS.south, CAMPUS_BOUNDS.west),
    createLatLng(kakao, CAMPUS_BOUNDS.north, CAMPUS_BOUNDS.east),
  );
}

function clampToCampus(kakao, lat, lng) {
  return createLatLng(
    kakao,
    clampCoordinate(lat, CAMPUS_BOUNDS.south, CAMPUS_BOUNDS.north),
    clampCoordinate(lng, CAMPUS_BOUNDS.west, CAMPUS_BOUNDS.east),
  );
}

function getKakaoMapLink(building) {
  return (
    building.placeUrl ??
    `https://map.kakao.com/link/map/${encodeURIComponent(building.name)},${building.lat},${building.lng}`
  );
}

function loadKakaoMap(appKey) {
  const hasPlaceholderKey =
    !appKey || appKey === "replace_with_your_kakao_javascript_key";

  if (typeof window === "undefined") {
    return Promise.reject(new Error("브라우저 환경에서만 지도를 불러올 수 있습니다."));
  }

  if (window.kakao?.maps) {
    return new Promise((resolve) => {
      window.kakao.maps.load(() => resolve(window.kakao));
    });
  }

  if (hasPlaceholderKey) {
    return Promise.reject(
      new Error("NEXT_PUBLIC_KAKAOMAP_KEY가 설정되지 않았습니다."),
    );
  }

  if (!kakaoMapLoaderPromise) {
    kakaoMapLoaderPromise = new Promise((resolve, reject) => {
      const existingScript = document.getElementById(KAKAO_MAP_SCRIPT_ID);

      const handleLoad = () => {
        window.kakao.maps.load(() => resolve(window.kakao));
      };

      const handleError = () => {
        reject(new Error("Kakao Maps SDK를 불러오지 못했습니다."));
      };

      if (existingScript) {
        existingScript.addEventListener("load", handleLoad, { once: true });
        existingScript.addEventListener("error", handleError, { once: true });
        return;
      }

      const script = document.createElement("script");
      script.id = KAKAO_MAP_SCRIPT_ID;
      script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${appKey}&autoload=false`;
      script.async = true;
      script.addEventListener("load", handleLoad, { once: true });
      script.addEventListener("error", handleError, { once: true });
      document.head.appendChild(script);
    }).catch((error) => {
      kakaoMapLoaderPromise = null;
      throw error;
    });
  }

  return kakaoMapLoaderPromise;
}

function createMarkerContent(building, onSelect) {
  const wrapper = document.createElement("button");
  wrapper.type = "button";
  wrapper.setAttribute("aria-label", `${building.name} 선택`);
  wrapper.style.display = "flex";
  wrapper.style.flexDirection = "column";
  wrapper.style.alignItems = "center";
  wrapper.style.gap = "8px";
  wrapper.style.cursor = "pointer";
  wrapper.style.border = "0";
  wrapper.style.padding = "0";
  wrapper.style.background = "transparent";
  wrapper.style.userSelect = "none";
  wrapper.style.transformOrigin = "center bottom";
  wrapper.style.transition = "transform 180ms ease, opacity 180ms ease";

  const pin = document.createElement("span");
  pin.style.display = "flex";
  pin.style.height = "42px";
  pin.style.width = "42px";
  pin.style.alignItems = "center";
  pin.style.justifyContent = "center";
  pin.style.borderRadius = "999px";
  pin.style.border = "3px solid rgba(255,255,255,0.96)";
  pin.style.background = building.markerColor;
  pin.style.boxShadow = "0 10px 24px rgba(15, 23, 42, 0.18)";
  pin.style.transition =
    "transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease";

  const dot = document.createElement("span");
  dot.style.height = "12px";
  dot.style.width = "12px";
  dot.style.borderRadius = "999px";
  dot.style.background = building.markerAccent;
  pin.appendChild(dot);

  const label = document.createElement("span");
  label.textContent = building.shortName;
  label.style.maxWidth = "118px";
  label.style.padding = "7px 11px";
  label.style.borderRadius = "999px";
  label.style.background = "rgba(255,255,255,0.96)";
  label.style.boxShadow = "0 8px 18px rgba(15, 23, 42, 0.12)";
  label.style.color = "#1f2937";
  label.style.fontSize = "11px";
  label.style.fontWeight = "700";
  label.style.lineHeight = "1.35";
  label.style.textAlign = "center";
  label.style.wordBreak = "keep-all";
  label.style.overflowWrap = "anywhere";
  label.style.transition =
    "background-color 180ms ease, color 180ms ease, transform 180ms ease";

  const handleClick = (event) => {
    event.preventDefault();
    event.stopPropagation();
    onSelect(building);
  };

  wrapper.addEventListener("click", handleClick);
  wrapper.append(pin, label);

  return { wrapper, pin, label };
}

function updateMarkerStyles(markerMap, activeId, visibleIds) {
  markerMap.forEach((record, buildingId) => {
    const isVisible = visibleIds.has(buildingId);
    const isActive = buildingId === activeId;

    record.wrapper.style.display = isVisible ? "flex" : "none";
    record.wrapper.style.opacity = isVisible ? "1" : "0";
    record.wrapper.style.transform = isActive
      ? "translateY(-3px)"
      : "translateY(0)";
    record.wrapper.style.zIndex = isActive ? "20" : "8";

    record.pin.style.transform = isActive ? "scale(1.08)" : "scale(1)";
    record.pin.style.borderColor = isActive
      ? record.building.markerAccent
      : "rgba(255,255,255,0.96)";
    record.pin.style.boxShadow = isActive
      ? "0 16px 32px rgba(15, 23, 42, 0.22)"
      : "0 10px 24px rgba(15, 23, 42, 0.18)";

    record.label.style.background = isActive
      ? "rgba(0, 56, 118, 0.94)"
      : "rgba(255,255,255,0.96)";
    record.label.style.color = isActive ? "#ffffff" : "#1f2937";
    record.label.style.transform = isActive ? "translateY(1px)" : "translateY(0)";
  });
}

function SearchBar({
  searchTerm,
  onChange,
  onClear,
  onSubmit,
  activeTag,
  onTagChange,
  tagOptions,
  filteredCount,
  totalCount,
  t,
}) {
  return (
    <form onSubmit={onSubmit} className={`${CARD_CLASS} p-4 sm:p-5`}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
            {t("map.ui.search")}
          </p>
          <p className="mt-2 break-keep text-sm leading-6 text-slate-500">
            {filteredCount} / {totalCount}{t("map.ui.showingBuildings")}
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-full bg-[#eef3fb] px-3 py-2 text-xs font-semibold text-[#4d6688]">
          <MapPin className="h-3.5 w-3.5 text-[#003876]" />
          {t("map.ui.coordinateApplied")}
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3 rounded-[22px] border border-[#d2dceb] bg-white px-4 py-3">
        <Search className="h-4 w-4 shrink-0 text-[#71839d]" />
        <input
          value={searchTerm}
          onChange={(event) => onChange(event.target.value)}
          placeholder={t("map.ui.searchPlaceholder")}
          className="min-w-0 flex-1 bg-transparent text-sm font-medium text-[#173050] outline-none placeholder:text-[#8a9ab1]"
        />

        {searchTerm ? (
          <button
            type="button"
            onClick={onClear}
            aria-label="검색어 지우기"
            className="flex h-8 w-8 items-center justify-center rounded-full bg-[#eef3fb] text-[#5a6d88] transition hover:bg-[#dde8f5]"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
      </div>

      <div className="mt-4">
        <div className="mb-2 flex items-center justify-between gap-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
            {t("map.ui.categoryFilter")}
          </p>
          <p className="text-xs font-medium text-slate-500">
            {activeTag === ALL_BUILDING_TAG ? t("map.filter.all") : t(activeTag)}
          </p>
        </div>

        <div className="flex gap-2 overflow-x-auto pb-1">
          {tagOptions.map((tag) => {
            const isActive = activeTag === tag;

            return (
              <button
                key={tag}
                type="button"
                onClick={() => onTagChange(tag)}
                aria-pressed={isActive}
                className={`shrink-0 rounded-full px-3 py-2 text-xs font-semibold transition ${
                  isActive
                    ? "bg-[#003876] text-white shadow-[0_10px_20px_rgba(0,56,118,0.18)]"
                    : "bg-white text-[#4d6688] hover:bg-[#eef3fb]"
                }`}
              >
                {t(tag)}
              </button>
            );
          })}
        </div>
      </div>
    </form>
  );
}

function SelectedBuildingCard({ building, onLocate, t }) {
  if (!building) {
    return (
      <section className={`${CARD_CLASS} p-5 sm:p-6`}>
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
          <Info className="h-4 w-4 text-[#003876]" />
          선택된 건물
        </div>
        <p className="mt-4 break-keep text-sm leading-7 text-slate-600">
          검색 결과가 없습니다. 다른 키워드로 다시 검색해보세요.
        </p>
      </section>
    );
  }

  return (
    <section className={`${CARD_CLASS} p-5 sm:p-6`}>
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
            <Building2 className="h-4 w-4 text-[#003876]" />
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
              <Building2 className="h-4 w-4 text-[#003876]" />
              <span className="whitespace-nowrap">
                {t("map.ui.selectedBuilding")}
              </span>
            </div>
          </div>

          <h2 className="mt-4 whitespace-nowrap text-2xl font-bold text-slate-900">
            {t(building.name)}
          </h2>

          <div className="mt-4 flex flex-wrap gap-2">
            <span className="rounded-full bg-[#e9f0fb] px-3 py-1 text-xs font-semibold text-[#003876]">
              {t(building.tag)}
            </span>
            <span className="rounded-full border border-[#d5deec] bg-white px-3 py-1 text-xs font-medium text-[#4f6483]">
              {t(building.summary)}
            </span>
          </div>
        </div>

        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#e9f0fb] text-[#003876]">
          <MapPinned className="h-5 w-5" />
        </div>
      </div>

      <div className="mt-5 rounded-[24px] border border-[#d5deec] bg-[#edf3fb] p-4">
        <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-[#003876]">
          <Route className="h-3.5 w-3.5" />
          {t("map.ui.officialLocationGuide")}
        </p>
        <p className="mt-2 break-keep text-sm leading-6 text-slate-700">
          {t(building.officialGuide)}
        </p>
      </div>

      <p className="mt-4 break-keep text-sm leading-7 text-slate-600">
        {t(building.description)}
      </p>

      <div className="mt-4 rounded-[22px] border border-[#d5deec] bg-[#f5f8fc] px-4 py-3 text-xs font-medium leading-6 text-[#5f718c]">
        <p>{t(building.kakaoPlaceName)}</p>
        <p className="mt-1">
          {t("map.ui.latitude")} {formatCoordinate(building.lat)} · {t("map.ui.longitude")} {formatCoordinate(building.lng)}
        </p>
      </div>

      <div className="mt-5 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={() => onLocate(building)}
          className={ACTION_BUTTON_CLASS}
        >
          <Compass className="h-4 w-4" />
          {t("map.ui.viewOnMap")}
        </button>

        <a
          href={getKakaoMapLink(building)}
          target="_blank"
          rel="noreferrer noopener"
          className={ACTION_BUTTON_CLASS}
        >
          <ArrowUpRight className="h-4 w-4" />
          {t("map.ui.openKakao")}
        </a>
      </div>
    </section>
  );
}

function BuildingListCard({
  buildings,
  selectedBuildingId,
  onSelect,
  isSearching,
  t,
}) {
  return (
    <section className={`${CARD_CLASS} p-4 sm:p-5`}>
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
        <MapPinned className="h-4 w-4 text-[#003876]" />
        {t("map.ui.buildingList")}
      </div>

      <p className="mt-2 break-keep text-sm leading-6 text-slate-500">
  {isSearching
    ? `${t("map.ui.searchResult")} ${buildings.length}${t("map.ui.countUnit")}`
    : `${t("map.ui.realCoordinateBuildings")} ${buildings.length}${t("map.ui.countUnit")}`}
</p>

      <div className="mt-4 space-y-3">
        {buildings.length > 0 ? (
          buildings.map((building) => {
            const isSelected = building.id === selectedBuildingId;

            return (
              <button
                key={building.id}
                type="button"
                onClick={() => onSelect(building)}
                className={`flex w-full items-start justify-between gap-3 rounded-[24px] border px-4 py-4 text-left transition ${
                  isSelected
                    ? "border-[#8ea8cf] bg-[#eaf1fb] shadow-[0_12px_28px_rgba(0,56,118,0.12)]"
                    : "border-[#d5deec] bg-white hover:bg-[#f5f8fc]"
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap gap-2">
                    <span className="rounded-full bg-[#f5f8fc] px-2.5 py-1 text-[11px] font-semibold text-[#5a6d88]">
                      {t(building.tag)}
                    </span>
                  </div>

                  <p className="mt-3 break-keep text-sm font-semibold text-slate-900">
                    {t(building.name)}
                  </p>

                  <p className="mt-1 break-keep text-xs leading-6 text-slate-500">
                    {t(building.officialGuide)}
                  </p>
                </div>

                <span
                  className={`mt-1 rounded-full p-2 ${
                    isSelected
                      ? "bg-[#d8e4f4] text-[#003876]"
                      : "bg-[#eef3fb] text-[#6a7f9c]"
                  }`}
                >
                  <ArrowUpRight className="h-4 w-4" />
                </span>
              </button>
            );
          })
        ) : (
          <div className="rounded-[24px] border border-dashed border-[#bcc8da] bg-[#f7faff] px-4 py-6 text-sm leading-7 text-[#5f718c]">
            {t("map.ui.noResult")}
          </div>
        )}
      </div>
    </section>
  );
}

function SourceCard({ t }) {
  return (
    <section className={`${CARD_CLASS} p-5 sm:p-6`}>
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
        <Info className="h-4 w-4 text-[#003876]" />
        {t("map.ui.dataSource")}
      </div>

      <p className="mt-4 break-keep text-sm leading-7 text-slate-600">
        {t(campusMapData.coordinateNote)}
      </p>

      <div className="mt-5 space-y-3">
        <a
          href={campusMapData.sourceUrl}
          target="_blank"
          rel="noreferrer noopener"
          className={`${ACTION_BUTTON_CLASS} w-full`}
        >
          <ArrowUpRight className="h-4 w-4" />
          {t(campusMapData.sourceLabel)}
        </a>

        <a
          href={campusMapData.coordinateSourceUrl}
          target="_blank"
          rel="noreferrer noopener"
          className={`${ACTION_BUTTON_CLASS} w-full`}
        >
          <ArrowUpRight className="h-4 w-4" />
          {t(campusMapData.coordinateSourceLabel)}
        </a>
      </div>
    </section>
  );
}

function MobilePanelTabs({ activeTab, onChange, t }) {
  return (
    <section className={`${CARD_CLASS} p-2`}>
      <div className="grid grid-cols-3 gap-2">
        {MOBILE_PANEL_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onChange(tab.id)}
              aria-pressed={isActive}
              className={`flex items-center justify-center gap-2 rounded-[20px] px-3 py-3 text-xs font-semibold transition ${
                isActive
                  ? "bg-[#003876] text-white shadow-[0_12px_24px_rgba(0,56,118,0.18)]"
                    : "bg-white text-[#4d6688] hover:bg-[#eef3fb]"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="whitespace-nowrap">
                {t(`map.ui.${tab.id}`)}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default function MapPage() {
  const { t } = useLanguage();
  const mapElementRef = useRef(null);
  const mapRef = useRef(null);
  const kakaoRef = useRef(null);
  const markerMapRef = useRef(new Map());
  const currentLocationMarkerRef = useRef(null);
  const currentLocationCircleRef = useRef(null);
  const selectBuildingRef = useRef(() => {});

  const [isMapReady, setIsMapReady] = useState(false);
  const [isLocating, setIsLocating] = useState(false);
  const [mapError, setMapError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTag, setActiveTag] = useState(ALL_BUILDING_TAG);
  const [mobilePanelTab, setMobilePanelTab] = useState(MOBILE_PANEL_TABS[0].id);
  const [selectedBuildingId, setSelectedBuildingId] = useState(
    buildingData[0]?.id ?? null,
  );

  const normalizedSearchTerm = searchTerm.trim().toLowerCase();
  const filteredBuildings = buildingData.filter(
    (building) =>
      matchesTag(building, activeTag) &&
      matchesSearch(building, normalizedSearchTerm, t),
  );

  const resolvedSelectedBuildingId = filteredBuildings.some(
    (building) => building.id === selectedBuildingId,
  )
    ? selectedBuildingId
    : filteredBuildings[0]?.id ?? null;

  const activeBuilding =
    filteredBuildings.find((building) => building.id === resolvedSelectedBuildingId) ??
    filteredBuildings[0] ??
    null;

  function panToBuilding(building) {
    if (!building || !mapRef.current || !kakaoRef.current) {
      return;
    }

    mapRef.current.panTo(createLatLng(kakaoRef.current, building.lat, building.lng));
  }

  function selectBuilding(building, { pan = true, revealOnMobile = true } = {}) {
    if (!building) {
      return;
    }

    setSelectedBuildingId(building.id);
    setMapError("");

    // 모바일에서는 카드가 길게 누적되기 쉬워서,
    // 사용자가 건물을 선택하면 상세 패널 탭으로 자연스럽게 이동시킵니다.
    if (revealOnMobile) {
      setMobilePanelTab("selected");
    }

    if (pan) {
      panToBuilding(building);
    }
  }

  useEffect(() => {
    selectBuildingRef.current = selectBuilding;
  });

  useEffect(() => {
    const visibleIds = new Set(filteredBuildings.map((building) => building.id));
    updateMarkerStyles(markerMapRef.current, resolvedSelectedBuildingId, visibleIds);
  }, [filteredBuildings, resolvedSelectedBuildingId]);

  useEffect(() => {
    let isMounted = true;
    let resizeHandler;

    async function initializeMap() {
      try {
        const kakao = await loadKakaoMap(process.env.NEXT_PUBLIC_KAKAOMAP_KEY);

        if (!isMounted || !mapElementRef.current) {
          return;
        }

        kakaoRef.current = kakao;

        // 지도의 최초 범위를 실제 건물 좌표 기준으로 맞춰서,
        // 화면에 보이는 건물 위치와 마커 위치가 어긋나지 않도록 시작합니다.
        const map = new kakao.maps.Map(mapElementRef.current, {
          center: createLatLng(
            kakao,
            campusMapData.defaultCenter.lat,
            campusMapData.defaultCenter.lng,
          ),
          level: campusMapData.zoom.defaultLevel,
        });

        map.setMinLevel(campusMapData.zoom.minLevel);
        map.setMaxLevel(campusMapData.zoom.maxLevel);
        map.setBounds(createCampusBounds(kakao), 48, 32, 48, 32);

        // 사용자가 지나치게 바깥으로 이동하지 않도록 중심 좌표를 캠퍼스 범위 안으로 보정합니다.
        const keepMapInsideCampus = () => {
          const center = map.getCenter();
          const lat = center.getLat();
          const lng = center.getLng();

          if (isInsideCampus(lat, lng)) {
            return;
          }

          map.setCenter(clampToCampus(kakao, lat, lng));
        };

        kakao.maps.event.addListener(map, "dragend", keepMapInsideCampus);
        kakao.maps.event.addListener(map, "idle", keepMapInsideCampus);

        // 반응형 레이아웃 전환 시 지도 타일이 깨지지 않도록 relayout을 호출합니다.
        resizeHandler = () => {
          map.relayout();
        };
        window.addEventListener("resize", resizeHandler);

        markerMapRef.current = new Map();

        // 마커는 건물 데이터만 보고 다시 렌더링할 수 있도록 DOM 생성과 상태 적용을 분리했습니다.
        buildingData.forEach((building) => {
          const markerContent = createMarkerContent(building, (selected) => {
            kakao.maps.event.preventMap();
            selectBuildingRef.current(selected);
          });

          const overlay = new kakao.maps.CustomOverlay({
            position: createLatLng(kakao, building.lat, building.lng),
            content: markerContent.wrapper,
            xAnchor: 0.5,
            yAnchor: 1,
            zIndex: 8,
          });

          overlay.setMap(map);
          markerMapRef.current.set(building.id, {
            building,
            overlay,
            ...markerContent,
          });
        });

        mapRef.current = map;
        setIsMapReady(true);
      } catch (error) {
        setMapError(error.message);
      }
    }

    initializeMap();

    return () => {
      isMounted = false;

      markerMapRef.current.forEach(({ overlay }) => overlay.setMap(null));
      markerMapRef.current = new Map();

      if (currentLocationMarkerRef.current) {
        currentLocationMarkerRef.current.setMap(null);
        currentLocationMarkerRef.current = null;
      }

      if (currentLocationCircleRef.current) {
        currentLocationCircleRef.current.setMap(null);
        currentLocationCircleRef.current = null;
      }

      if (resizeHandler) {
        window.removeEventListener("resize", resizeHandler);
      }
    };
  }, []);

  function handleSearchSubmit(event) {
    event.preventDefault();

    if (filteredBuildings.length > 0) {
      selectBuilding(filteredBuildings[0]);
    }
  }

  function handleMoveToCurrentLocation() {
    if (!navigator.geolocation) {
      setMapError("이 브라우저에서는 현재 위치 기능을 사용할 수 없습니다.");
      return;
    }

    if (!mapRef.current || !kakaoRef.current) {
      setMapError("지도를 아직 불러오는 중입니다. 잠시 후 다시 시도해주세요.");
      return;
    }

    setIsLocating(true);
    setMapError("");

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const kakao = kakaoRef.current;

        if (!isInsideCampus(latitude, longitude)) {
          setMapError("현재 위치가 경기대학교 수원캠퍼스 표시 범위를 벗어나 있습니다.");
          setIsLocating(false);
          return;
        }

        const currentPosition = createLatLng(kakao, latitude, longitude);

        if (!currentLocationMarkerRef.current) {
          currentLocationMarkerRef.current = new kakao.maps.Marker({
            position: currentPosition,
            title: "현재 위치",
          });
        }

        currentLocationMarkerRef.current.setPosition(currentPosition);
        currentLocationMarkerRef.current.setMap(mapRef.current);

        if (currentLocationCircleRef.current) {
          currentLocationCircleRef.current.setMap(null);
        }

        currentLocationCircleRef.current = new kakao.maps.Circle({
          center: currentPosition,
          radius: 45,
          strokeWeight: 2,
          strokeColor: "#0f766e",
          strokeOpacity: 0.9,
          fillColor: "#2dd4bf",
          fillOpacity: 0.25,
        });

        currentLocationCircleRef.current.setMap(mapRef.current);
        mapRef.current.panTo(currentPosition);
        setIsLocating(false);
      },
      (error) => {
        const errorMessage =
          error.code === error.PERMISSION_DENIED
            ? "위치 권한이 거부되었습니다. 브라우저 설정을 확인해주세요."
            : "현재 위치를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.";

        setMapError(errorMessage);
        setIsLocating(false);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
      },
    );
  }

  return (
              <main className="min-h-screen bg-[linear-gradient(180deg,#d7deea_0%,#c6c9d4_100%)] text-slate-900">
                <div className="mx-auto w-full max-w-[430px] px-4 py-4">
                  <header className="overflow-hidden rounded-[32px] bg-gradient-to-r from-[#dae2ef] via-[#d1d9e6] to-[#c3d0e2] p-4 shadow-[0_22px_55px_rgba(0,56,118,0.08)] sm:p-7">
                    <div className="flex flex-col gap-5">
                      <div className="min-w-0">
      
                        <div className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-2 text-xs font-semibold text-[#003876]">
                          <MapPinned className="h-3.5 w-3.5 text-[#003876]" />
                          {t("map.ui.badge")}
                        </div>

                        <h1 className="mt-4 break-keep text-[30px] font-bold text-[#173050] sm:text-4xl">
                          {t(campusMapData.campusName)}
                        </h1>

                        <p className="mt-3 break-keep text-sm leading-6 text-[#4d6688] sm:hidden">
                          {t("map.ui.mobileDescription")}
                        </p>

                        <p className="mt-3 break-keep text-sm leading-6 text-[#4d6688]">
                          {t(campusMapData.description)}
                        </p>

                        <div className="mt-4 flex flex-wrap gap-2 sm:mt-5">
                        

                          <span className="rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-[#003876]">
                            Kakao Maps API
                          </span>

                          <span className="rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-[#003876]">
                            {t("map.ui.coordinate")}
                          </span>
                        </div>

                      <div className="mt-4 rounded-[22px] border border-white/70 bg-white/50 px-4 py-3 backdrop-blur sm:hidden">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#003876]">
                          {t("map.ui.campusInfo")}
                        </p>
                        <p className="mt-2 break-keep text-sm leading-6 text-[#4d6688]">
                          {t(campusMapData.address)}
                        </p>
                      </div>
                    </div>

                    <div className="hidden">
                      <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#003876]">
                        {t("map.ui.campusInfo")}
                      </p>

                      <p className="mt-3 break-keep text-sm leading-6 text-[#4d6688]">
                        {t(campusMapData.address)}
                      </p>

                      <p className="mt-3 break-keep text-xs leading-6 text-[#5f718c]">
                        {t(campusMapData.coordinateNote)}
                      </p>

                      <a
                        href={campusMapData.sourceUrl}
                        target="_blank"
                        rel="noreferrer noopener"
                        className={`${ACTION_BUTTON_CLASS} mt-4`}
                      >
                      <ArrowUpRight className="h-4 w-4" />
                        {t("map.ui.official")}
                      </a>
                    </div>

                   </div>
                  </header>

        <div className="mt-5 grid gap-5">
          <section className="min-w-0 space-y-5">
            <SearchBar
              searchTerm={searchTerm}
              onChange={setSearchTerm}
              onClear={() => setSearchTerm("")}
              onSubmit={handleSearchSubmit}
              activeTag={activeTag}
              onTagChange={setActiveTag}
              tagOptions={BUILDING_TAG_OPTIONS}
              filteredCount={filteredBuildings.length}
              totalCount={buildingData.length}
              t={t}
            />

            <div className={`${CARD_CLASS} p-3 sm:p-4`}>
              <div className="flex flex-col gap-3 border-b border-[#efe6db] px-2 pb-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
                    {t("map.ui.campusMap")}
                  </p>
                  <p className="mt-2 break-keep text-sm leading-6 text-slate-500">
                    {t("map.ui.markerGuide")}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2">
  <div className="inline-flex items-center gap-2 rounded-full bg-[#eef3fb] px-3 py-2 text-xs font-semibold text-[#4d6688]">
    <MapPin className="h-3.5 w-3.5 text-[#003876]" />
    <span className="break-keep">
   {t("map.ui.selectedBuilding")} {activeBuilding ? t(activeBuilding.shortName) : t("map.ui.none")}
</span>
  </div>

  <div className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-2 text-xs font-semibold text-[#4d6688]">
    <Building2 className="h-3.5 w-3.5 text-[#003876]" />
    <span className="break-keep">
  {t("map.ui.filter")} {activeTag === ALL_BUILDING_TAG ? t("map.filter.all") : t(activeTag)}
</span>
  </div>
</div>
              </div>

              <div className="relative mt-4 overflow-hidden rounded-[26px] border border-[#bfcde1] bg-[#d7deea]">
                <div
                  ref={mapElementRef}
                  className="h-[46svh] min-h-[330px] w-full max-h-[540px] bg-[#d7deea] sm:h-[560px] sm:max-h-none lg:h-[720px]"
                  aria-label="경기대학교 수원캠퍼스 지도"
                />

                <div className="pointer-events-none absolute left-4 top-4 max-w-[240px]">
                  <div className="rounded-[18px] border border-[#d2dceb] bg-white/94 px-3 py-2 text-xs font-semibold leading-5 text-[#4d6688] shadow-sm">
                    {t("map.ui.coordinateNotice")}
                  </div>
                </div>

                {filteredBuildings.length === 0 ? (
                  <div className="pointer-events-none absolute inset-x-4 top-20 rounded-[20px] border border-[#d2dceb] bg-white/95 px-4 py-3 text-sm font-medium leading-6 text-[#516782] shadow-[0_14px_26px_rgba(0,56,118,0.1)]">
                    {t("map.ui.noMatchingBuilding")}
                  </div>
                ) : null}

                <div className="absolute bottom-4 right-4 z-10">
                  <button
                    type="button"
                    onClick={handleMoveToCurrentLocation}
                    disabled={!isMapReady || isLocating}
                    className={`${ACTION_BUTTON_CLASS} shadow-[0_16px_35px_rgba(0,56,118,0.12)] disabled:cursor-not-allowed disabled:opacity-60`}
                  >
                    <Locate className="h-4 w-4" />
                    <span>{isLocating ? t("map.ui.locating") : t("map.ui.currentLocation")}</span>
                  </button>
                </div>

                {!isMapReady && !mapError ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-[#d7deea]/70 backdrop-blur-[2px]">
                    <div className="rounded-full border border-[#d2dceb] bg-white/96 px-5 py-3 text-sm font-semibold text-[#4d6688] shadow-lg">
                      {t("map.ui.loading")}
                    </div>
                  </div>
                ) : null}
              </div>

              {mapError ? (
                <div className="mt-4 rounded-[22px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium leading-6 text-rose-600">
                  {mapError}
                </div>
              ) : null}
            </div>

            <div className="space-y-4">
              <MobilePanelTabs
                activeTab={mobilePanelTab}
                onChange={setMobilePanelTab}
                t={t}
              />

              {mobilePanelTab === "selected" ? (
                <SelectedBuildingCard
                  building={activeBuilding}
                  onLocate={(building) => selectBuilding(building)}
                  t={t}
                />
              ) : null}

              {mobilePanelTab === "list" ? (
                <BuildingListCard
                  buildings={filteredBuildings}
                  selectedBuildingId={resolvedSelectedBuildingId}
                  onSelect={(building) => selectBuilding(building)}
                  isSearching={
                    Boolean(normalizedSearchTerm) ||
                    activeTag !== ALL_BUILDING_TAG
                  }
                  t={t}
                />
              ) : null}

              {mobilePanelTab === "source" ? <SourceCard t={t} /> : null}
            </div>
          </section>

          <aside className="hidden">
            <SelectedBuildingCard
              building={activeBuilding}
              onLocate={(building) => selectBuilding(building)}
              t={t}
            />
            <BuildingListCard
              buildings={filteredBuildings}
              selectedBuildingId={resolvedSelectedBuildingId}
              onSelect={(building) => selectBuilding(building)}
              isSearching={
                Boolean(normalizedSearchTerm) || activeTag !== ALL_BUILDING_TAG
              }
              t={t}
            />
            <SourceCard t={t} />
          </aside>
        </div>
      </div>
    </main>
  );
}
