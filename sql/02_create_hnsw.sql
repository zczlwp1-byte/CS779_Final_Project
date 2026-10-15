DROP INDEX IF EXISTS papers_embedding_hnsw_idx;

CREATE INDEX papers_embedding_hnsw_idx
ON papers
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);

ANALYZE papers;