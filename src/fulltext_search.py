import argparse
from time import perf_counter

import psycopg
from psycopg.types.json import Jsonb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default="reducing the computational cost of transformer models",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    print(f"Query: {args.query}")

    with psycopg.connect(
        host="localhost",
        port=5433,
        dbname="cs779_vector_db",
        user="postgres",
        password="cs779password",
    ) as connection:
        with connection.cursor() as cursor:
            start_time = perf_counter()

            cursor.execute(
                """
                WITH search_query AS (
                    SELECT websearch_to_tsquery(
                        'english', %s
                    ) AS query
                )
                SELECT
                    arxiv_id,
                    title,
                    abstract,
                    category,
                    ts_rank_cd(
                        search_document,
                        search_query.query
                    ) AS rank
                FROM papers, search_query
                WHERE search_document @@ search_query.query
                ORDER BY rank DESC
                LIMIT %s;
                """,
                (args.query, args.limit),
            )

            results = cursor.fetchall()
            latency_ms = (
                perf_counter() - start_time
            ) * 1000

            history_results = [
                {
                    "arxiv_id": row[0],
                    "title": row[1],
                    "rank": float(row[4]),
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
                    "full_text",
                    latency_ms,
                    Jsonb(history_results),
                ),
            )

        connection.commit()

    print(f"\nDatabase search latency: {latency_ms:.3f} ms")
    print(f"Results returned: {len(results)}\n")

    for rank_number, result in enumerate(results, start=1):
        arxiv_id, title, abstract, category, text_rank = result
        preview = abstract[:220].replace("\n", " ")

        print(f"{rank_number}. {title}")
        print(f"   arXiv ID: {arxiv_id}")
        print(f"   Category: {category}")
        print(f"   Text rank: {text_rank:.4f}")
        print(f"   Abstract: {preview}...")
        print()


if __name__ == "__main__":
    main()