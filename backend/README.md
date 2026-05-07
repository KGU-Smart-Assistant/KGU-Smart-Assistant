# KGU Smart Assistant Backend

## 실행 방법

`backend` 폴더에서 실행합니다.

```powershell
docker compose up -d --build
```

프론트엔드까지 같이 실행하려면 폴더 구조가 아래처럼 되어 있어야 합니다.

```text
KGU SmartAssistant/
  backend/
    docker-compose.yml
  frontend/
    frontend-app/
```

`docker-compose.yml`은 프론트엔드를 `../frontend/frontend-app` 경로에서 찾습니다.

## 로컬에 생성되는 파일/폴더

실행하거나 크롤링하면 아래 폴더가 로컬에 생길 수 있습니다.

```text
backend/.tmp/
backend/.crawl4ai-data/
```

용도:

- `.tmp/`: 크롤링 로그, ingest report, 임시 실행 결과 저장
- `.crawl4ai-data/`: Crawl4AI/브라우저 실행 데이터 저장

Docker volume으로는 아래 데이터가 생성됩니다.

```text
postgres_data
chroma_data
frontend_node_modules
frontend_next
```

이 데이터들도 로컬 실행 데이터입니다.

## 접속 주소

- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:8000
- Swagger 문서: http://localhost:8000/docs
- Chroma: http://localhost:8001
