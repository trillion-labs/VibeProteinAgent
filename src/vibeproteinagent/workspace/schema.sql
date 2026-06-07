-- VibeProteinAgent workspace SQLite schema (Architecture v1 §5 + §9).
-- Self-applicable to a fresh empty file; safe to re-apply.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
  job_id            TEXT PRIMARY KEY,
  call_id           TEXT UNIQUE,
  state             TEXT NOT NULL CHECK (state IN ('submitted','queued','running','completed','failed','cancelled')),
  tool_name         TEXT NOT NULL,
  tool_version      TEXT NOT NULL,
  params_json       TEXT NOT NULL,
  params_hash       TEXT NOT NULL,
  output_json       TEXT,
  error_code        TEXT,
  error_message     TEXT,
  error_details_json TEXT,
  log_path          TEXT NOT NULL,
  workspace         TEXT NOT NULL,
  trajectory_id     TEXT,
  workflow_id       TEXT,
  workflow_node_id  TEXT,
  worker_id         INTEGER,
  gpu_ids           TEXT,
  priority          TEXT NOT NULL DEFAULT 'normal' CHECK (priority IN ('high','normal','low')),
  retry_count       INTEGER NOT NULL DEFAULT 0,
  max_retries       INTEGER NOT NULL DEFAULT 0,
  submitted_at      INTEGER NOT NULL,
  queued_at         INTEGER,
  started_at        INTEGER,
  finished_at       INTEGER,
  cache_hit         INTEGER NOT NULL DEFAULT 0,
  cancel_requested  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_state     ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_tool      ON jobs(tool_name, tool_version);
CREATE INDEX IF NOT EXISTS idx_jobs_traj      ON jobs(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_jobs_wf        ON jobs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_jobs_paramhash ON jobs(tool_name, tool_version, params_hash);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id    TEXT PRIMARY KEY,
  sha256         TEXT NOT NULL,
  kind           TEXT NOT NULL,
  mime           TEXT,
  size_bytes     INTEGER NOT NULL,
  pool_path      TEXT NOT NULL,
  producer_job   TEXT,
  created_at     INTEGER NOT NULL,
  last_used_at   INTEGER NOT NULL,
  pinned         INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (producer_job) REFERENCES jobs(job_id) ON DELETE SET NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(sha256);
CREATE INDEX        IF NOT EXISTS idx_artifacts_lru ON artifacts(pinned, last_used_at);

CREATE TABLE IF NOT EXISTS workflows (
  workflow_id    TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  version        TEXT NOT NULL,
  descriptor     TEXT NOT NULL,
  params_json    TEXT NOT NULL,
  trajectory_id  TEXT,
  state          TEXT NOT NULL CHECK (state IN ('submitted','running','completed','failed','cancelled')),
  submitted_at   INTEGER NOT NULL,
  finished_at    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_workflows_traj ON workflows(trajectory_id);

CREATE TABLE IF NOT EXISTS trajectories (
  trajectory_id       TEXT PRIMARY KEY,
  workspace           TEXT NOT NULL,
  user_query          TEXT NOT NULL,
  objective           TEXT,
  state               TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open','completed')),
  final_submit_job_id TEXT,
  report_path         TEXT,
  created_at          INTEGER NOT NULL,
  updated_at          INTEGER NOT NULL,
  completed_at        INTEGER
);

CREATE TABLE IF NOT EXISTS trajectory_reports (
  trajectory_id        TEXT PRIMARY KEY,
  report_path          TEXT NOT NULL,
  source_id            TEXT NOT NULL,
  format               TEXT NOT NULL,
  final_submit_job_id  TEXT,
  created_at           INTEGER NOT NULL,
  FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_nodes (
  workflow_id    TEXT NOT NULL,
  node_id        TEXT NOT NULL,
  tool_name      TEXT NOT NULL,
  tool_version   TEXT NOT NULL,
  upstream       TEXT NOT NULL,
  params_hash    TEXT NOT NULL,
  input_hash     TEXT NOT NULL,
  cache_key      TEXT NOT NULL,
  state          TEXT NOT NULL CHECK (state IN ('pending','ready','running','completed','failed','cached','skipped')),
  job_id         TEXT,
  output_json    TEXT,
  error_code     TEXT,
  error_message  TEXT,
  error_details_json TEXT,
  PRIMARY KEY (workflow_id, node_id),
  FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
  FOREIGN KEY (job_id)      REFERENCES jobs(job_id)           ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_wfn_cache ON workflow_nodes(cache_key);

CREATE TABLE IF NOT EXISTS runs (
  run_id          TEXT PRIMARY KEY,
  job_id          TEXT NOT NULL,
  attempt         INTEGER NOT NULL,
  worker_id       INTEGER NOT NULL,
  gpu_ids         TEXT NOT NULL,
  started_at      INTEGER NOT NULL,
  finished_at     INTEGER,
  exit_state      TEXT,
  vram_peak_mib   INTEGER,
  duration_ms     INTEGER,
  FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job_id);

CREATE TABLE IF NOT EXISTS job_artifacts (
  job_id       TEXT NOT NULL,
  field        TEXT NOT NULL,
  artifact_id  TEXT NOT NULL,
  PRIMARY KEY (job_id, field),
  FOREIGN KEY (job_id)      REFERENCES jobs(job_id)           ON DELETE CASCADE,
  FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
