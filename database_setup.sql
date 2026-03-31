-- Run this in MySQL Workbench before starting SynthEye with MySQL settings.
-- You can change DB/user/password as needed.

CREATE DATABASE IF NOT EXISTS syntheye
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Optional: dedicated app user (recommended over root in production)
-- CREATE USER IF NOT EXISTS 'syntheye_app'@'%' IDENTIFIED BY 'replace_with_strong_password';
-- GRANT ALL PRIVILEGES ON syntheye.* TO 'syntheye_app'@'%';
-- FLUSH PRIVILEGES;

-- Table creation is automatic when api_server.py starts.
