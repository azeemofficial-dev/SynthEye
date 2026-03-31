# SynthEye Monitoring and Backup Plan

## Runtime Monitoring
- Health endpoint: `GET /api/health`
- Metrics endpoint: `GET /api/metrics`
- DB status endpoint: `GET /api/db/stats`

Recommended alert conditions:
- `status != "ok"` from `/api/health`
- `database.ready == false`
- `requests_error_total` spikes in `/api/metrics`
- `rate_limited_total` spikes in `/api/metrics`
- low disk space for `logs/` and backup folders

## Logging
- App logs are written to:
  - container stdout/stderr
  - rotating file: `logs/syntheye.log` (`SYNTHEYE_LOG_FILE`)
- Rotation policy in backend:
  - max size: 5 MB per file
  - keep last 5 files

Recommended:
- ship container logs to your platform logging (CloudWatch, ELK, Datadog)
- retain logs for at least 14 days

## Backups
- Use MySQL backup scripts:
  - Linux/macOS: `ops/backup_mysql.sh`
  - Windows: `ops/backup_mysql.ps1`
- Default retention in scripts: 14 days

Recommended schedule:
- daily full logical backup
- weekly restore test into a staging database

## Disaster Recovery Checklist
1. Restore latest SQL dump into MySQL.
2. Re-deploy app containers.
3. Verify `/api/health` and `/api/db/stats`.
4. Validate login and one analysis request.
