import argparse
from time import perf_counter

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default="reducing the computational cost of transformers",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    print(f"Query: {args.query}")
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    query_embedding = model.encode(
        args.query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    with psycopg.connect(
        host="localhost",
        port=5433,
        dbname="cs779_vector_db",
        user="postgres",
        password="cs779password",
    ) as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            start_time = perf_counter()

            cursor.execute(
                """
                SELECT
                    arxiv_id,
                    title,
                    abstract,
                    category,
                    1 - (embedding <=> %s) AS similarity
                FROM papers
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (
                    query_embedding,
                    query_embedding,
                    args.limit,
                ),
            )

            results = cursor.fetchall()
            latency_ms = (
                perf_counter() - start_time
            ) * 1000

            history_results = [
                {
                    "arxiv_id": row[0],
                    "title": row[1],
                    "similarity": float(row[4]),
                }
                for row in results
            ]

            cursor.execute(
                """
                INSERT INTO query_history (
                    query_text,
                    search_method,
                    latency_ms,
                    retrieved_papers
                )
                VALUES (%s, %s, %s, %s);
                """,
                (
                    args.query,
                    "exact_vector",
                    latency_ms,
                    Jsonb(history_results),
                ),
            )

        connection.commit()

    print(f"\nDatabase search latency: {latency_ms:.3f} ms")
    print(f"Top {len(results)} results:\n")

    for rank, result in enumerate(results, start=1):
        arxiv_id, title, abstract, category, similarity = result
        preview = abstract[:220].replace("\n", " ")

        print(f"{rank}. {title}")
        print(f"   arXiv ID: {arxiv_id}")
        print(f"   Category: {category}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Abstract: {preview}...")
        print()


if __name__ == "__main__":
    main()