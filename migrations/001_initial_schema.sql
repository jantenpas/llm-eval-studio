-- 001: Full schema — projects, test_suites, test_cases, runs, results
-- Drops legacy tables (runs, results) and recreates everything to spec.

DROP TABLE IF EXISTS results;
DROP TABLE IF EXISTS runs;

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    endpoint_url TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS test_suites (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_test_suites_project_id ON test_suites(project_id);

CREATE TABLE IF NOT EXISTS test_cases (
    id              TEXT PRIMARY KEY,
    suite_id        TEXT NOT NULL REFERENCES test_suites(id),
    input           TEXT NOT NULL,
    expected_output TEXT NOT NULL,
    scoring_config  TEXT,
    scoring_method  TEXT NOT NULL DEFAULT 'exact_match',
    tags            TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_test_cases_suite_id ON test_cases(suite_id);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id),
    suite_id      TEXT NOT NULL REFERENCES test_suites(id),
    name          TEXT NOT NULL,
    llm_model     TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    pass_threshold REAL NOT NULL DEFAULT 0.70,
    status        TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    avg_score     REAL,
    passed        INTEGER,
    created_at    TEXT NOT NULL,
    completed_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);

CREATE TABLE IF NOT EXISTS results (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    test_case_id    TEXT NOT NULL REFERENCES test_cases(id),
    input           TEXT NOT NULL,
    expected_output TEXT NOT NULL,
    scoring_config  TEXT,
    actual_output   TEXT NOT NULL,
    scoring_method  TEXT NOT NULL,
    score           REAL NOT NULL,
    passed          INTEGER NOT NULL,
    latency_ms      INTEGER NOT NULL,
    reasoning       TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_test_case_id ON results(test_case_id);
