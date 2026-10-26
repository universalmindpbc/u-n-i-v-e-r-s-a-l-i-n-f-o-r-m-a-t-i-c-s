-- Additive, versioned evidence memory; existing literature tables stay intact.
CREATE TABLE IF NOT EXISTS research.agent_experiments (
 id text NOT NULL, revision text NOT NULL, question text NOT NULL,
 data jsonb NOT NULL, json_text text NOT NULL, csv_text text NOT NULL,
 json_sha256 text NOT NULL, csv_sha256 text NOT NULL,
 json_path text NOT NULL, csv_path text NOT NULL,
 recorded_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(id,revision)
);
CREATE TABLE IF NOT EXISTS research.agent_memory (
 id text NOT NULL, revision text NOT NULL,
 kind text NOT NULL CHECK(kind IN ('experiment','mission','research_direction')),
 data jsonb NOT NULL, document text NOT NULL, source_sha256 text NOT NULL,
 embedding vector(512),
 search tsvector GENERATED ALWAYS AS (to_tsvector('english',document)) STORED,
 PRIMARY KEY(id,revision)
);
CREATE INDEX IF NOT EXISTS agent_memory_search ON research.agent_memory USING gin(search);
GRANT SELECT ON research.agent_experiments,research.agent_memory TO informatics_reader;
