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
  state          TEXT NOT NULL CHECK (state IN ('submitted','running','completed','failed','cancelled')),
  submitted_at   INTEGER NOT NULL,
  finished_at    INTEGER
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

CREATE TABLE IF NOT EXISTS agent_runs (
  run_id                 TEXT PRIMARY KEY,
  workspace              TEXT NOT NULL,
  user_request           TEXT NOT NULL,
  mode                   TEXT,
  state                  TEXT NOT NULL,
  provider               TEXT,
  model                  TEXT,
  provider_snapshot_json TEXT,
  system_prompt_version  TEXT NOT NULL,
  tool_schema_version    TEXT NOT NULL,
  classification_json    TEXT,
  workflow_ids_json      TEXT NOT NULL DEFAULT '[]',
  job_ids_json           TEXT NOT NULL DEFAULT '[]',
  artifacts_json         TEXT NOT NULL DEFAULT '[]',
  final_report_json      TEXT,
  created_at             INTEGER NOT NULL,
  updated_at             INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at);

CREATE TABLE IF NOT EXISTS agent_messages (
  message_id   TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL,
  role         TEXT NOT NULL,
  content      TEXT NOT NULL,
  model        TEXT,
  token_count  INTEGER,
  created_at   INTEGER NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_agent_messages_run ON agent_messages(run_id);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
  tool_call_id TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL,
  tool_name    TEXT NOT NULL,
  args_json    TEXT NOT NULL,
  result_json  TEXT,
  ok           INTEGER NOT NULL,
  created_at   INTEGER NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run ON agent_tool_calls(run_id);

CREATE TABLE IF NOT EXISTS agent_plans (
  plan_id                TEXT PRIMARY KEY,
  run_id                 TEXT NOT NULL,
  plan_json              TEXT NOT NULL,
  status                 TEXT NOT NULL,
  validation_errors_json TEXT NOT NULL DEFAULT '[]',
  workflow_id            TEXT,
  created_at             INTEGER NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_summaries (
  summary_id   TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL,
  workflow_id  TEXT,
  job_id        TEXT,
  summary_json  TEXT NOT NULL,
  created_at    INTEGER NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_decisions (
  decision_id   TEXT PRIMARY KEY,
  run_id        TEXT NOT NULL,
  round_index   INTEGER NOT NULL,
  decision_json TEXT NOT NULL,
  created_at    INTEGER NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_faithfulness_checks (
  check_id      TEXT PRIMARY KEY,
  run_id        TEXT NOT NULL,
  round_index   INTEGER NOT NULL,
  decision_id   TEXT,
  result_json   TEXT NOT NULL,
  created_at    INTEGER NOT NULL,
  FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_agent_faithfulness_run ON agent_faithfulness_checks(run_id);
