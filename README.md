# Semantic Search for Academic Papers

CS779 Vector Database Final Project

This project implements an end-to-end academic-paper retrieval system with PostgreSQL and pgvector. It downloads arXiv metadata, generates 384-dimensional embeddings, compares exact search with HNSW and IVFFlat, evaluates PostgreSQL full-text search, and demonstrates local retrieval-augmented generation (RAG) with numbered paper citations.

## Research Questions

1. How do HNSW and IVFFlat compare with exact vector search in latency, Recall@10, build time, and index size?
2. How does semantic retrieval compare with PostgreSQL full-text retrieval?
3. How does performance change from 5,000 to 10,000 and 30,000 papers?

## Features

- checkpointed arXiv data collection
- all-MiniLM-L6-v2 embeddings with 384 dimensions
- PostgreSQL 16 and pgvector 0.8.6
- exact cosine, HNSW, IVFFlat, and GIN full-text search
- mean latency, P95, and Recall@10 benchmarks
- local FLAN-T5 Base RAG generation
- numbered citations and JSONB query history

## Architecture

~~~text
arXiv API
  -> clean and deduplicate
  -> generate MiniLM embeddings
  -> store in PostgreSQL + pgvector
  -> retrieve with Exact / HNSW / IVFFlat / full-text
  -> send Top-5 abstracts to FLAN-T5
  -> validate citations and log query history
~~~

## Project Structure

~~~text
CS779_Final_Project/
|-- README.md
|-- compose.yaml
|-- requirements.txt
|-- data/processed/
|-- sql/
|   |-- 01_create_tables.sql
|   |-- 02_create_hnsw.sql
|   +-- 03_create_ivfflat.sql
|-- src/
|   |-- test_setup.py
|   |-- prepare_data.py
|   |-- generate_embeddings.py
|   |-- load_database.py
|   |-- semantic_search.py
|   |-- fulltext_search.py
|   |-- benchmark_search.py
|   |-- benchmark_fulltext.py
|   +-- rag_search.py
|-- results/
+-- deliverables/
~~~

The data CSV and NPY files are generated artifacts. If submission size is limited, omit them and retain the scripts plus benchmark result CSV files.

## Prerequisites

- Windows 10 or 11
- Docker Desktop with WSL 2
- Python 3.11 or later
- PowerShell
- approximately 3 GB of available disk space

## Setup

Open PowerShell in the project directory.

### Create and activate the virtual environment

~~~powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
~~~

### Install dependencies

~~~powershell
python -m pip install --upgrade pip
pip install -r .\requirements.txt
pip install transformers sentencepiece
~~~

Required packages include feedparser, numpy, pandas, pgvector, psycopg[binary], requests, sentence-transformers, sentencepiece, torch, and transformers.

### Start and verify PostgreSQL

~~~powershell
docker compose up -d
docker ps
python .\src\test_setup.py
~~~

Development configuration:

~~~text
Container: cs779-postgres
Host: localhost
Port: 5433
Database: cs779_vector_db
User: postgres
~~~

The password in the scripts is for local coursework only. Production code should use environment variables.

## Data Pipeline

### Prepare 30,000 papers

~~~powershell
python .\src\prepare_data.py --count 30000
~~~

The downloader saves checkpoints and uses daily date windows when the arXiv API rejects offsets beyond 10,000. Rerun the same command after HTTP 429, 500, or 503 to resume.

Validate the CSV:

~~~powershell
python -c "import pandas as pd; df=pd.read_csv(r'data\processed\arxiv_30000.csv'); print('Rows:', len(df)); print(df.isna().sum()); print('Duplicates:', df['arxiv_id'].duplicated().sum())"
~~~

### Generate embeddings

~~~powershell
python .\src\generate_embeddings.py --count 30000
~~~

Validate the array:

~~~powershell
python -c "import numpy as np; x=np.load(r'data\processed\arxiv_30000_embeddings.npy'); print(x.shape, x.dtype); print('NaN:', np.isnan(x).sum()); print('Infinite:', np.isinf(x).sum())"
~~~

Expected shape is (30000, 384), dtype is float32, and invalid-value counts are zero.

### Load the database

~~~powershell
docker exec cs779-postgres psql -U postgres -d cs779_vector_db -c "DROP INDEX IF EXISTS papers_embedding_hnsw_idx; DROP INDEX IF EXISTS papers_embedding_ivfflat_idx;"
python .\src\load_database.py --count 30000
~~~

Validate row counts:

~~~powershell
docker exec cs779-postgres psql -U postgres -d cs779_vector_db -c "SELECT COUNT(*) AS total_papers, COUNT(embedding) AS papers_with_embeddings, COUNT(search_document) AS papers_with_search_document FROM papers;"
~~~

All three values should be 30,000.

## ANN Indexes

Keep only the ANN index required for the current controlled test.

### HNSW

~~~powershell
docker exec cs779-postgres psql -U postgres -d cs779_vector_db -c "DROP INDEX IF EXISTS papers_embedding_hnsw_idx; DROP INDEX IF EXISTS papers_embedding_ivfflat_idx;"

$hnswBuild = Measure-Command {
    Get-Content -Raw .\sql\02_create_hnsw.sql |
    docker exec -i cs779-postgres psql -U postgres -d cs779_vector_db
}

$hnswBuild.TotalMilliseconds
docker exec cs779-postgres psql -U postgres -d cs779_vector_db -c "\di+ papers_embedding_hnsw_idx"
~~~

### IVFFlat

~~~powershell
docker exec cs779-postgres psql -U postgres -d cs779_vector_db -c "DROP INDEX IF EXISTS papers_embedding_hnsw_idx; DROP INDEX IF EXISTS papers_embedding_ivfflat_idx;"

$ivfflatBuild = Measure-Command {
    Get-Content -Raw .\sql\03_create_ivfflat.sql |
    docker exec -i cs779-postgres psql -U postgres -d cs779_vector_db
}

$ivfflatBuild.TotalMilliseconds
docker exec cs779-postgres psql -U postgres -d cs779_vector_db -c "\di+ papers_embedding_ivfflat_idx"
~~~

## Search

~~~powershell
python .\src\semantic_search.py "database indexing and query optimization" --limit 5
python .\src\fulltext_search.py "database indexing and query optimization" --limit 5
~~~

## Benchmarks

Protocol:

- 10 fixed queries
- Top 10 results
- three warm-up runs
- 10 measured runs
- exact cosine search as ANN ground truth
- HNSW ef_search = 40
- IVFFlat probes = 8

Run after creating the corresponding index:

~~~powershell
python .\src\benchmark_search.py --method hnsw --count 30000
python .\src\benchmark_search.py --method ivfflat --count 30000
python .\src\benchmark_fulltext.py --count 30000
~~~

Raw and summary CSV files are written to results. Full-text overlap measures agreement with exact semantic Top 10 results; it is not a human relevance score.

## RAG Demonstration

~~~powershell
python .\src\rag_search.py "How can neural networks be trained more efficiently?" --method ivfflat --limit 5
~~~

The first run downloads approximately 1 GB of FLAN-T5 files. Later runs use the local cache. The selected HNSW or IVFFlat index must exist before its method is used.

Verify history:

~~~powershell
docker exec cs779-postgres psql -U postgres -d cs779_vector_db -c "SELECT query_text, search_method, ROUND(latency_ms::numeric, 3) AS latency_ms, created_at FROM query_history ORDER BY created_at DESC LIMIT 5;"
~~~

## Results

### Vector Scaling

| Corpus | Exact mean | HNSW mean | HNSW Recall@10 | IVFFlat mean | IVFFlat Recall@10 |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 9.867 ms | 1.045 ms | 98% | 0.953 ms | 90% |
| 10,000 | 18.903 ms | 1.036 ms | 96% | 1.091 ms | 92% |
| 30,000 | 63.385 ms | 1.123 ms | 91% | 2.268 ms | 93% |

At 30K, HNSW was 56.46 times faster than its exact control. IVFFlat was 28.45 times faster than its exact control. HNSW had the lowest ANN latency; IVFFlat provided two additional recall percentage points, a faster build, and a smaller index.

### 30K Index Construction

| Index | Build time | Size |
|---|---:|---:|
| HNSW | 4.695 s | 59 MB |
| IVFFlat | 1.493 s | 47 MB |

### Full-Text Scaling

| Corpus | Mean latency | Mean P95 | Mean results | Exact-vector overlap |
|---:|---:|---:|---:|---:|
| 5,000 | 1.029 ms | 1.204 ms | 6.3 / 10 | 11% |
| 10,000 | 1.226 ms | 1.410 ms | 8.0 / 10 | 13% |
| 30,000 | 1.816 ms | 2.042 ms | 9.0 / 10 | 10% |

### Representative RAG Run

| Component | Result |
|---|---:|
| IVFFlat Top-5 retrieval | 2.449 ms |
| Local FLAN-T5 generation | 1,846.560 ms |
| Retrieval plus generation | 1,849.009 ms |
| Citations | Sources [5] and [3] |
| History method | rag_ivfflat |

Model loading is excluded. Local CPU generation dominates the response time.

## Expected Warnings

These warnings are benign when the run completes:

- unauthenticated Hugging Face requests
- unavailable Windows symlinks
- FLAN-T5 tied-weights warning
- HNSW maintenance_work_mem warning at 30K

Do not set tie_word_embeddings to False. That can cause missing T5 embedding weights and unusable output.

## Troubleshooting

### Python for-loop error in PowerShell

Python code was pasted directly into PowerShell. Save it in a .py file and run the file.

### arXiv fails after offset 10,000

Rerun prepare_data.py with the same count. It resumes from the checkpoint and uses daily windows.

### Full-text latency is about 70 ms

Confirm the GIN index exists and that bitmap scans are enabled after validation.

### RAG retrieval takes hundreds of milliseconds

Use only vector distance in SQL ordering:

~~~sql
ORDER BY embedding <=> %s
~~~

A secondary arxiv_id sort can prevent efficient ANN index ordering. The final script also performs three warm-up runs.

### RAG returns only three dots

Load FLAN-T5 with its original configuration and do not override tie_word_embeddings.

## Limitations

- The corpus is category-based, not a random sample of all arXiv.
- Human relevance was reviewed for only a limited query set.
- Exact-vector overlap is not an independent relevance metric.
- Long inputs may be truncated.
- Experiments used CPU inference on one machine.
- FLAN-T5 may produce short answers, so citation validation and an extractive fallback are used.
- Development credentials remain in the example scripts.

## Final Submission Checklist

- [ ] README.md is in the project root.
- [ ] requirements.txt includes every dependency.
- [ ] Python files pass syntax checks.
- [ ] SQL scripts and compose.yaml are included.
- [ ] Result CSV files exist for 5K, 10K, and 30K.
- [ ] The representative RAG command produces citations.
- [ ] Report and presentation are in deliverables.
- [ ] Virtual environments, caches, checkpoints, and temporary files are excluded.
- [ ] Large generated data files are included only if required.
- [ ] Passwords and local paths are reviewed before public sharing.

Recommended ZIP:

~~~text
README.md
requirements.txt
compose.yaml
src/
sql/
results/
deliverables/
~~~

Exclude:

~~~text
.venv/
__pycache__/
.git/
.idea/
.vscode/
Hugging Face cache
temporary checkpoints
~~~

## References

- [arXiv API User's Manual](https://info.arxiv.org/help/api/user-manual.html)
- [pgvector documentation](https://github.com/pgvector/pgvector)
- [Sentence Transformers semantic search documentation](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [FLAN-T5 Base model card](https://huggingface.co/google/flan-t5-base)
