import torch
import sentence_transformers
import psycopg

print("PyTorch version:", torch.__version__)
print("Sentence Transformers version:", sentence_transformers.__version__)

with psycopg.connect(
    host="localhost",
    port=5433,
    dbname="cs779_vector_db",
    user="postgres",
    password="cs779password"
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
        )
        vector_version = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' "
            "AND table_name IN ('papers', 'query_history');"
        )
        table_count = cursor.fetchone()[0]

print("pgvector version:", vector_version)
print("Required database tables:", table_count)
print("Setup test passed.")