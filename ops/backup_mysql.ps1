param(
  [string]$EnvFile = ".env.production",
  [string]$BackupDir = "backups\\mysql",
  [int]$RetentionDays = 14
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path $EnvFile) {
  Get-Content $EnvFile | ForEach-Object {
    if ($_ -match "^\s*#" -or $_ -match "^\s*$") { return }
    $parts = $_.Split("=", 2)
    if ($parts.Length -eq 2) {
      [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
    }
  }
}

$mysqlUser = [System.Environment]::GetEnvironmentVariable("MYSQL_USER")
$mysqlPassword = [System.Environment]::GetEnvironmentVariable("MYSQL_PASSWORD")
$mysqlDatabase = [System.Environment]::GetEnvironmentVariable("MYSQL_DATABASE")

if (-not $mysqlUser -or -not $mysqlPassword -or -not $mysqlDatabase) {
  throw "MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE are required in $EnvFile"
}

$fullBackupDir = Join-Path $root $BackupDir
New-Item -ItemType Directory -Path $fullBackupDir -Force | Out-Null
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$outFile = Join-Path $fullBackupDir "syntheye_${timestamp}.sql"

Write-Host "Creating backup at $outFile"
$dumpCmd = "mysqldump -u`"$mysqlUser`" -p`"$mysqlPassword`" `"$mysqlDatabase`" --single-transaction --quick --lock-tables=false"
& docker compose -f docker-compose.prod.yml exec -T mysql sh -lc $dumpCmd | Out-File -FilePath $outFile -Encoding utf8

Get-ChildItem -Path $fullBackupDir -Filter "*.sql" |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetentionDays) } |
  Remove-Item -Force

Write-Host "Backup complete. Retention: $RetentionDays days."
