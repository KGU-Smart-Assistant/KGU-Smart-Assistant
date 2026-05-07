param(
    [string]$OutputDir = ".tmp\volume-export",
    [string]$PostgresContainer = "postgres-db",
    [string]$ChromaVolume = "backend_chroma_data",
    [string]$DbUser = "myuser",
    [string]$DbName = "mydatabase"
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

docker exec $PostgresContainer pg_dump -U $DbUser -d $DbName -Fc -f /tmp/kgusmart_postgres.dump
docker cp "${PostgresContainer}:/tmp/kgusmart_postgres.dump" "$OutputDir\kgusmart_postgres.dump"

docker run --rm `
    -v "${ChromaVolume}:/volume-data" `
    -v "${PWD}\${OutputDir}:/backup" `
    alpine sh -c "cd /volume-data && tar -czf /backup/chroma_data.tar.gz ."

Write-Host "Exported PostgreSQL and Chroma data to $OutputDir"
