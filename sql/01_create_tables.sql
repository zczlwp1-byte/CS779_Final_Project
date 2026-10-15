CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    paper_id BIGSERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    category TEXT,
    publication_date DATE,
    embedding vector(384),
    search_document tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(abstract, '')), 'B')
    ) STORED
);

CREATE INDEX IF NOT EXISTS papers_search_document_gin
ON papers
USING GIN (search_document);

CREATE TABLE IF NOT EXISTS query_history (
    query_id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    search_method TEXT NOT NULL,
    latency_ms DOUBLE PRECISION,
    retrieved_papers JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);