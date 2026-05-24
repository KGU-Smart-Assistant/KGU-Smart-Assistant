# KGU Smart Assistant Frontend Migration

공식 레포 `KGU-Smart-Assistant/Frontend`의 `develop` 브랜치 구조를 기준으로 현재 프로젝트를 맞춘 버전입니다.

## 현재 구조

```text
C:\kgu\캡스톤
├─ app
│  ├─ globals.css
│  ├─ layout.jsx
│  ├─ page.jsx
│  ├─ map
│  │  └─ page.jsx
│  └─ phone
│     └─ page.jsx
├─ components
│  ├─ BottomNav.jsx
│  └─ Header.jsx
├─ data
│  └─ mapData.js
├─ .env.example
├─ .env.local
├─ AGENTS.md
├─ CLAUDE.md
├─ eslint.config.mjs
├─ jsconfig.json
├─ next.config.mjs
├─ package-lock.json
├─ package.json
├─ postcss.config.mjs
└─ README.md
```

## 마이그레이션 반영 내용

- `next.config.mjs`에 `@/` alias를 추가했습니다.
- `jsconfig.json`에 `@/*` 경로 설정을 추가했습니다.
- `app/layout.jsx`에서 공통 `Header`, `BottomNav`를 렌더링하도록 변경했습니다.
- `components/Header.jsx`, `components/BottomNav.jsx`를 공식 구조 기준으로 추가했습니다.
- `app/phone/page.jsx`를 새로 추가했습니다.
- `CLAUDE.md`, `AGENTS.md`를 추가했습니다.
- `app/map/page.jsx`의 데이터 import를 `@/data/mapData`로 통일했습니다.

## 공통 레이아웃

`app/layout.jsx`는 아래 역할을 담당합니다.

- `./globals.css` import
- `@/components/Header` 렌더링
- `children` 렌더링
- `@/components/BottomNav` 렌더링
- `paddingBottom: "80px"` 적용

따라서 각 페이지에서 `Header`, `BottomNav`를 중복으로 렌더링하지 않아야 합니다.

## 경로 alias 사용법

공유 모듈은 `@/` 기준으로 import합니다.

예시:

```jsx
import Header from "@/components/Header";
import BottomNav from "@/components/BottomNav";
import { buildingData, campusMapData } from "@/data/mapData";
```

같은 폴더 내부 파일은 기존처럼 상대경로를 유지합니다.

예시:

```jsx
import "./globals.css";
```

## 실행 방법

### 1. 의존성 설치

```powershell
npm.cmd install
```

### 2. 개발 서버 실행

```powershell
npm.cmd run dev
```

### 3. 브라우저 접속

- 홈: [http://localhost:3000](http://localhost:3000)
- 지도: [http://localhost:3000/map](http://localhost:3000/map)
- 전화번호: [http://localhost:3000/phone](http://localhost:3000/phone)

## 카카오맵 환경변수

`.env.local`에 아래 값을 넣어야 지도 페이지가 정상 동작합니다.

```env
NEXT_PUBLIC_KAKAOMAP_KEY=여기에_카카오_JavaScript_Key
```

## 카카오 Developers 설정

아래 두 가지가 빠지면 지도 SDK가 정상 로드되지 않을 수 있습니다.

- 카카오맵 사용 설정을 `ON`
- Web 플랫폼에 `http://localhost:3000` 등록

## 검증 명령

```powershell
npm.cmd run lint
npm.cmd run build
```

## 기존 GitHub 저장소에 반영할 때

기존 공식 프론트엔드 저장소 안에 넣을 때는 보통 아래 파일들을 우선 비교/병합하면 됩니다.

- `app/layout.jsx`
- `app/page.jsx`
- `app/map/page.jsx`
- `app/phone/page.jsx`
- `components/Header.jsx`
- `components/BottomNav.jsx`
- `data/mapData.js`
- `next.config.mjs`
- `jsconfig.json`
- `AGENTS.md`
- `CLAUDE.md`

## GitHub에 올리면 안 되는 파일

- `.env.local`
- `.next/`
- `node_modules/`
