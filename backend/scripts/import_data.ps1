param(
    [string]$InputDir = ".tmp\volume-export",
    [string]$PostgresContainer = "postgres-db",
    [string]$ChromaVolume = "backend_chroma_data",
    [string]$DbUser = "myuser",
    [string]$DbName = "mydatabase"
)

docker cp "$InputDir\kgusmart_postgres.dump" "${PostgresContainer}:/tmp/kgusmart_postgres.dump"
docker exec $PostgresContainer pg_restore -U $DbUser -d $DbName --clean --if-exists /tmp/kgusmart_postgres.dump

docker compose stop chroma
docker run --rm `
    -v "${ChromaVolume}:/volume-data" `
    -v "${PWD}\${InputDir}:/backup" `
    alpine sh -c "rm -rf /volume-data/* && tar -xzf /backup/chroma_data.tar.gz -C /volume-data"
docker compose up -d

Write-Host "Imported PostgreSQL and Chroma data from $InputDir"
