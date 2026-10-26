-- Dedicated Informatics staging database. No private I Am or Re-Model tables.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS research;
CREATE TABLE IF NOT EXISTS research.publications (
 id text PRIMARY KEY, data jsonb NOT NULL, document text NOT NULL,
 search tsvector GENERATED ALWAYS AS (to_tsvector('english', document)) STORED,
 version text NOT NULL
);
CREATE INDEX IF NOT EXISTS publications_search ON research.publications USING gin(search);
CREATE TABLE IF NOT EXISTS research.passages (
 id text PRIMARY KEY, publication_id text NOT NULL REFERENCES research.publications(id),
 kind text NOT NULL CHECK (kind IN ('approved-record','full-text')),
 section text NOT NULL, text text NOT NULL, source_url text NOT NULL,
 rights jsonb NOT NULL, sha256 text NOT NULL, embedding vector(512),
 search tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
 version text NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_search ON research.passages USING gin(search);
CREATE INDEX IF NOT EXISTS passages_publication ON research.passages(publication_id);
CREATE TABLE IF NOT EXISTS research.atlas (id text PRIMARY KEY, modulus integer UNIQUE, data jsonb NOT NULL, sha256 text NOT NULL);
CREATE TABLE IF NOT EXISTS research.releases (version text PRIMARY KEY, manifest jsonb NOT NULL);
