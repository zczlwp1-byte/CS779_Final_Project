import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "cs779_vector_db",
    "user": "postgres",
    "password": "cs779password",
}

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

TOP_K = 10
WARMUP_RUNS = 3
MEASURED_RUNS = 10

QUERIES = [
    (
        "reducing the computational cost "
        "of transformer models"
    ),
    (
        "database indexing and "
        "query optimization"
    ),
    (
        "image classification using "
        "deep neural networks"
    ),
    (
        "natural language processing "
        "with large language models"
    ),
    (
        "robot navigation and "
        "autonomous systems"
    ),
    (
        "information retrieval and "
        "semantic search"
    ),
    "cybersecurity threat detection",
    (
        "reinforcement learning for "
        "decision making"
    ),
    (
        "medical image analysis using "
        "artificial intelligence"
    ),
    (
        "efficient training of "
        "neural networks"
    ),
]

EXACT_SQL = """
SELECT arxiv_id
FROM papers
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s, arxiv_id
LIMIT %s;
"""

FULLTEXT_SQL = """
WITH search_query AS (
    SELECT websearch_to_tsquery(
        'english',
        %s
    ) AS query
)
SELECT
    arxiv_id,
    title,
    ts_rank_cd(
        search_document,
        search_query.query
    ) AS rank
FROM papers, search_query
WHERE search_document @@ search_query.query
ORDER BY rank DESC, arxiv_id
LIMIT %s;
"""


def validate_database(
    connection,
    expected_count,
):
    result = connection.execute(
        """
        SELECT
            COUNT(*) AS total_papers,
            COUNT(embedding) AS embeddings,
            COUNT(search_document) AS documents
        FROM papers;
        """
    ).fetchone()

    total_papers = result[0]
    embeddings = result[1]
    documents = result[2]

    if total_papers != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} papers, "
            f"but database contains "
            f"{total_papers}."
        )

    if embeddings != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} embeddings, "
            f"but found {embeddings}."
        )

    if documents != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} "
            f"search documents, "
            f"but found {documents}."
        )

    gin_index_exists = connection.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'papers'
              AND indexdef ILIKE '%USING gin%'
        );
        """
    ).fetchone()[0]

    if not gin_index_exists:
        raise RuntimeError(
            "No GIN full-text index was found "
            "on the papers table."
        )

    print(
        f"Database validation passed: "
        f"{total_papers} papers, "
        f"{embeddings} embeddings, "
        f"{documents} search documents."
    )

    print(
        "GIN full-text index validation passed."
    )


def get_exact_reference(
    connection,
    query_embedding,
):
    with connection.transaction():
        # Force PostgreSQL to calculate
        # the exact vector Top 10.
        connection.execute(
            "SET LOCAL enable_indexscan = off;"
        )
        connection.execute(
            "SET LOCAL enable_bitmapscan = off;"
        )
        connection.execute(
            "SET LOCAL enable_seqscan = on;"
        )

        rows = connection.execute(
            EXACT_SQL,
            (
                query_embedding,
                TOP_K,
            ),
        ).fetchall()

    return [
        row[0]
        for row in rows
    ]


def run_fulltext_search(
    connection,
    query,
):
    with connection.transaction():
        # Explicitly allow PostgreSQL to use
        # the GIN full-text index.
        connection.execute(
            "SET LOCAL enable_seqscan = on;"
        )
        connection.execute(
            "SET LOCAL enable_indexscan = on;"
        )
        connection.execute(
            "SET LOCAL enable_bitmapscan = on;"
        )

        start_time = perf_counter()

        rows = connection.execute(
            FULLTEXT_SQL,
            (
                query,
                TOP_K,
            ),
        ).fetchall()

        latency_ms = (
            perf_counter() - start_time
        ) * 1000

    paper_ids = [
        row[0]
        for row in rows
    ]

    return latency_ms, paper_ids


def calculate_overlap(
    exact_ids,
    fulltext_ids,
):
    exact_set = set(exact_ids)
    fulltext_set = set(fulltext_ids)

    if not exact_set:
        return 0.0

    overlap_count = len(
        exact_set.intersection(
            fulltext_set
        )
    )

    return overlap_count / len(exact_set)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        choices=[
            500,
            5000,
            10000,
            30000,
        ],
        required=True,
        help=(
            "Expected number of papers "
            "in the database."
        ),
    )

    args = parser.parse_args()

    print(
        f"Expected database rows: "
        f"{args.count}"
    )

    print(
        f"Loading model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Generating query embeddings..."
    )

    query_embeddings = model.encode(
        QUERIES,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    raw_results = []
    summary_results = []

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:
        register_vector(connection)

        validate_database(
            connection,
            args.count,
        )

        # End the validation transaction.
        # This prevents SET LOCAL settings
        # from leaking into later searches.
        connection.commit()

        for query_number, (
            query,
            embedding,
        ) in enumerate(
            zip(
                QUERIES,
                query_embeddings,
            ),
            start=1,
        ):
            print()
            print("=" * 70)

            print(
                f"Query {query_number}/"
                f"{len(QUERIES)}: "
                f"{query}"
            )

            print("=" * 70)

            exact_ids = get_exact_reference(
                connection,
                embedding,
            )

            print(
                "Testing full-text search..."
            )

            # Warm-up runs are not included
            # in the final statistics.
            for _ in range(
                WARMUP_RUNS
            ):
                run_fulltext_search(
                    connection,
                    query,
                )

            latencies = []
            result_counts = []
            overlaps = []

            for run_number in range(
                1,
                MEASURED_RUNS + 1,
            ):
                (
                    latency_ms,
                    fulltext_ids,
                ) = run_fulltext_search(
                    connection,
                    query,
                )

                result_count = len(
                    fulltext_ids
                )

                overlap = calculate_overlap(
                    exact_ids,
                    fulltext_ids,
                )

                latencies.append(
                    latency_ms
                )

                result_counts.append(
                    result_count
                )

                overlaps.append(
                    overlap
                )

                raw_results.append(
                    {
                        "query_number": (
                            query_number
                        ),
                        "query": query,
                        "method": (
                            "full_text"
                        ),
                        "run": run_number,
                        "latency_ms": (
                            latency_ms
                        ),
                        "results_returned": (
                            result_count
                        ),
                        (
                            "overlap_with_"
                            "exact_at_10"
                        ): overlap,
                    }
                )

            mean_latency = float(
                np.mean(latencies)
            )

            p95_latency = float(
                np.percentile(
                    latencies,
                    95,
                )
            )

            mean_results = float(
                np.mean(
                    result_counts
                )
            )

            mean_overlap = float(
                np.mean(overlaps)
            )

            summary_results.append(
                {
                    "query_number": (
                        query_number
                    ),
                    "query": query,
                    "method": "full_text",
                    "mean_latency_ms": (
                        mean_latency
                    ),
                    "p95_latency_ms": (
                        p95_latency
                    ),
                    "mean_results_returned": (
                        mean_results
                    ),
                    (
                        "overlap_with_"
                        "exact_at_10"
                    ): mean_overlap,
                }
            )

            print(
                f"  Mean latency:   "
                f"{mean_latency:.3f} ms"
            )

            print(
                f"  P95 latency:    "
                f"{p95_latency:.3f} ms"
            )

            print(
                f"  Results:        "
                f"{mean_results:.1f}/10"
            )

            print(
                f"  Exact overlap:  "
                f"{mean_overlap:.1%}"
            )

    results_directory = Path(
        "results"
    )

    results_directory.mkdir(
        exist_ok=True
    )

    raw_output = (
        results_directory
        / (
            f"benchmark_{args.count}_"
            f"fulltext_raw.csv"
        )
    )

    summary_output = (
        results_directory
        / (
            f"benchmark_{args.count}_"
            f"fulltext_summary.csv"
        )
    )

    raw_dataframe = pd.DataFrame(
        raw_results
    )

    summary_dataframe = pd.DataFrame(
        summary_results
    )

    raw_dataframe.to_csv(
        raw_output,
        index=False,
    )

    summary_dataframe.to_csv(
        summary_output,
        index=False,
    )

    overall = {
        "mean_latency_ms": (
            summary_dataframe[
                "mean_latency_ms"
            ].mean()
        ),
        "mean_p95_latency_ms": (
            summary_dataframe[
                "p95_latency_ms"
            ].mean()
        ),
        "mean_results_returned": (
            summary_dataframe[
                "mean_results_returned"
            ].mean()
        ),
        "mean_overlap_with_exact_at_10": (
            summary_dataframe[
                (
                    "overlap_with_"
                    "exact_at_10"
                )
            ].mean()
        ),
    }

    print()
    print("=" * 70)

    print(
        "OVERALL FULL-TEXT RESULTS"
    )

    print("=" * 70)

    print(
        f"Mean latency:      "
        f"{overall['mean_latency_ms']:.3f} ms"
    )

    print(
        f"Mean P95 latency:  "
        f"{overall['mean_p95_latency_ms']:.3f} ms"
    )

    print(
        f"Mean results:      "
        f"{overall['mean_results_returned']:.2f}/10"
    )

    print(
        f"Mean exact overlap: "
        f"{overall['mean_overlap_with_exact_at_10']:.1%}"
    )

    print()

    print(
        f"Raw results saved to: "
        f"{raw_output}"
    )

    print(
        f"Summary saved to: "
        f"{summary_output}"
    )

    print(
        "Full-text benchmark completed."
    )


if __name__ == "__main__":
    main()