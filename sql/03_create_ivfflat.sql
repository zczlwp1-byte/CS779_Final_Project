DROP INDEX IF EXISTS papers_embedding_hnsw_idx;
DROP INDEX IF EXISTS papers_embedding_ivfflat_idx;

CREATE INDEX papers_embedding_ivfflat_idx
ON papers
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 70);

ANALYZE papers;