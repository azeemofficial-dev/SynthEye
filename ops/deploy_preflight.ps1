param(
  [string]$EnvFile = ".env.production"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ([System.IO.Path]::IsPathRooted($EnvFile)) {
  $envPath = $EnvFile
} else {
  $envPath = Join-Path $root $EnvFile
}

$errors = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

function Add-Error([string]$message) {
  $errors.Add($message) | Out-Null
}

function Add-Warning([string]$message) {
  $warnings.Add($message) | Out-Null
}

function Read-EnvMap([string]$Path) {
  $map = @{}
  foreach ($lineRaw in Get-Content -LiteralPath $Path) {
    $line = $lineRaw.Trim()
    if (-not $line) {
      continue
    }
    if ($line.StartsWith("#")) {
      continue
    }
    $parts = $line.Split("=", 2)
    if ($parts.Length -ne 2) {
      continue
    }
    $key = $parts[0].Trim()
    $value = $parts[1].Trim()
    $map[$key] = $value
  }
  return $map
}

Write-Host "=== SynthEye Deploy Preflight ==="
Write-Host "Project root: $root"
Write-Host "Env file    : $envPath"
Write-Host ""

if (-not (Test-Path -LiteralPath $envPath)) {
  Add-Error "Missing env file: $envPath"
} else {
  $envMap = Read-EnvMap $envPath

  $required = @(
    "SYNTH_DOMAIN",
    "LETSENCRYPT_EMAIL",
    "MYSQL_ROOT_PASSWORD",
    "MYSQL_DATABASE",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "SYNTHEYE_ENV",
    "SYNTHEYE_REQUIRE_AUTH",
    "SYNTHEYE_SESSION_SECRET",
    "SYNTHEYE_SESSION_HTTPS_ONLY",
    "SYNTHEYE_ALLOW_ORIGINS",
    "SYNTHEYE_DB_HOST",
    "SYNTHEYE_DB_PORT",
    "SYNTHEYE_DB_USER",
    "SYNTHEYE_DB_PASSWORD",
    "SYNTHEYE_DB_NAME"
  )

  foreach ($key in $required) {
    if (-not $envMap.ContainsKey($key) -or -not $envMap[$key]) {
      Add-Error "Missing required env key: $key"
    }
  }

  if ($envMap.ContainsKey("SYNTHEYE_ENV") -and $envMap["SYNTHEYE_ENV"] -ne "production") {
    Add-Error "SYNTHEYE_ENV must be production in .env.production"
  }

  if ($envMap.ContainsKey("SYNTHEYE_SESSION_SECRET")) {
    $secret = $envMap["SYNTHEYE_SESSION_SECRET"]
    if ($secret.Length -lt 32) {
      Add-Error "SYNTHEYE_SESSION_SECRET must be at least 32 characters."
    }
    if ($secret -eq "dev-only-change-this-before-deploy") {
      Add-Error "SYNTHEYE_SESSION_SECRET uses the development default and must be replaced."
    }
  }

  if ($envMap.ContainsKey("SYNTH_DOMAIN")) {
    $domain = $envMap["SYNTH_DOMAIN"]
    if ($domain -match "localhost|127\.0\.0\.1") {
      Add-Error "SYNTH_DOMAIN cannot be localhost/127.0.0.1 for production TLS."
    }
    if ($domain -match "example\.com$|yourdomain|your-domain|change") {
      Add-Warning "SYNTH_DOMAIN looks like a placeholder. Use your real public domain."
    }
  }

  if ($envMap.ContainsKey("SYNTHEYE_ALLOW_ORIGINS")) {
    foreach ($origin in $envMap["SYNTHEYE_ALLOW_ORIGINS"].Split(",")) {
      $trimmed = $origin.Trim()
      if (-not $trimmed) {
        continue
      }
      if (-not $trimmed.StartsWith("https://")) {
        Add-Error "CORS origin must start with https:// in production: $trimmed"
      }
      if ($trimmed -match "localhost|127\.0\.0\.1") {
        Add-Error "CORS origin cannot include localhost/127.0.0.1 in production: $trimmed"
      }
    }
  }

  $placeholderPattern = "replace_with_|<password>|changeme|change-this|example\.com"
  foreach ($entry in $envMap.GetEnumerator()) {
    if ($entry.Value -match $placeholderPattern) {
      Add-Warning "Possible placeholder value in $($entry.Key)"
    }
  }
}

$modelPaths = @(
  (Join-Path $root "models\deepfake\deepfake_detector.keras"),
  (Join-Path $root "models\misinfo\vectorizer.joblib"),
  (Join-Path $root "models\misinfo\classifier.joblib")
)

foreach ($modelPath in $modelPaths) {
  if (-not (Test-Path -LiteralPath $modelPath)) {
    Add-Error "Missing model artifact: $modelPath"
  }
}

$dockerCmd = $null
$dockerFromPath = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerFromPath) {
  $dockerCmd = $dockerFromPath.Source
}
if (-not $dockerCmd) {
  $dockerDefaultPath = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
  if (Test-Path -LiteralPath $dockerDefaultPath) {
    $dockerCmd = $dockerDefaultPath
  }
}

if (-not $dockerCmd) {
  Add-Warning "Docker is not installed on this machine. Compose validation and deploy commands were skipped."
} else {
  & $dockerCmd compose version *> $null
  if ($LASTEXITCODE -ne 0) {
    Add-Warning "Docker CLI is installed but Compose/engine is not ready yet (often fixed after reboot and Docker Desktop start)."
  } else {
    if (Test-Path -LiteralPath $envPath) {
      & $dockerCmd compose -f (Join-Path $root "docker-compose.prod.yml") --env-file $envPath config -q
      if ($LASTEXITCODE -ne 0) {
        Add-Error "docker compose config validation failed."
      }
    }
  }
}

Write-Host ""
if ($warnings.Count -gt 0) {
  Write-Host "Warnings:"
  foreach ($w in $warnings) {
    Write-Host "  - $w"
  }
  Write-Host ""
}

if ($errors.Count -gt 0) {
  Write-Host "Preflight FAILED:"
  foreach ($e in $errors) {
    Write-Host "  - $e"
  }
  exit 1
}

Write-Host "Preflight PASSED."
Write-Host "You can deploy with:"
Write-Host "docker compose -f docker-compose.prod.yml --env-file $EnvFile up -d --build"
