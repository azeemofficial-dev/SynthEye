#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.production"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

BACKUP_DIR="${BACKUP_DIR:-${ROOT_DIR}/backups/mysql}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
mkdir -p "${BACKUP_DIR}"

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_FILE="${BACKUP_DIR}/syntheye_${TS}.sql.gz"

echo "Creating backup at ${OUT_FILE}"
docker compose -f "${ROOT_DIR}/docker-compose.prod.yml" exec -T mysql sh -lc \
  "mysqldump -u\"${MYSQL_USER}\" -p\"${MYSQL_PASSWORD}\" \"${MYSQL_DATABASE}\" --single-transaction --quick --lock-tables=false" \
  | gzip > "${OUT_FILE}"

find "${BACKUP_DIR}" -type f -name "*.sql.gz" -mtime "+${RETENTION_DAYS}" -delete
echo "Backup complete. Retention: ${RETENTION_DAYS} days."
