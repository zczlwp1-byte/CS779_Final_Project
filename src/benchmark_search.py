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

HNSW_EF_SEARCH = 40
IVFFLAT_PROBES = 8

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

SEARCH_SQL = """
SELECT arxiv_id, title
FROM papers
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s
LIMIT %s;
"""


def validate_database(
    connection,
    expected_count,
    method,
):
    row_count = connection.execute(
        "SELECT COUNT(*) FROM papers;"
    ).fetchone()[0]

    if row_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} papers, "
            f"but the database contains "
            f"{row_count}."
        )

    if method == "hnsw":
        selected_index = (
            "papers_embedding_hnsw_idx"
        )
        conflicting_index = (
            "papers_embedding_ivfflat_idx"
        )
    else:
        selected_index = (
            "papers_embedding_ivfflat_idx"
        )
        conflicting_index = (
            "papers_embedding_hnsw_idx"
        )

    selected_exists = connection.execute(
        """
        SELECT to_regclass(%s);
        """,
        (f"public.{selected_index}",),
    ).fetchone()[0]

    if selected_exists is None:
        raise RuntimeError(
            f"Required index "
            f"{selected_index} does not exist."
        )

    conflicting_exists = connection.execute(
        """
        SELECT to_regclass(%s);
        """,
        (f"public.{conflicting_index}",),
    ).fetchone()[0]

    if conflicting_exists is not None:
        raise RuntimeError(
            f"Conflicting index "
            f"{conflicting_index} still exists. "
            "Drop it before running this benchmark."
        )

    print(
        f"Database validation passed: "
        f"{row_count} papers."
    )
    print(
        f"Selected index: {selected_index}"
    )


def run_search(
    connection,
    query_embedding,
    method,
):
    with connection.transaction():
        if method == "exact":
            # Force an exact sequential scan.
            connection.execute(
                "SET LOCAL enable_indexscan = off;"
            )
            connection.execute(
                "SET LOCAL enable_bitmapscan = off;"
            )
            connection.execute(
                "SET LOCAL enable_seqscan = on;"
            )

        elif method == "hnsw":
            # Force use of the HNSW index.
            connection.execute(
                "SET LOCAL enable_seqscan = off;"
            )
            connection.execute(
                "SET LOCAL enable_indexscan = on;"
            )
            connection.execute(
                "SET LOCAL enable_bitmapscan = off;"
            )
            connection.execute(
                f"SET LOCAL hnsw.ef_search = "
                f"{HNSW_EF_SEARCH};"
            )

        elif method == "ivfflat":
            # Force use of the IVFFlat index.
            connection.execute(
                "SET LOCAL enable_seqscan = off;"
            )
            connection.execute(
                "SET LOCAL enable_indexscan = on;"
            )
            connection.execute(
                "SET LOCAL enable_bitmapscan = off;"
            )
            connection.execute(
                f"SET LOCAL ivfflat.probes = "
                f"{IVFFLAT_PROBES};"
            )

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

        start_time = perf_counter()

        rows = connection.execute(
            SEARCH_SQL,
            (
                query_embedding,
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


def calculate_recall(
    exact_ids,
    approximate_ids,
):
    exact_set = set(exact_ids)
    approximate_set = set(
        approximate_ids
    )

    overlap = len(
        exact_set.intersection(
            approximate_set
        )
    )

    if not exact_set:
        return 0.0

    return overlap / len(exact_set)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=[
            "hnsw",
            "ivfflat",
        ],
        required=True,
        help=(
            "Approximate vector index "
            "to benchmark."
        ),
    )

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
        f"Benchmark method: {args.method}"
    )
    print(
        f"Expected database rows: "
        f"{args.count}"
    )

    if args.method == "hnsw":
        print(
            f"HNSW ef_search: "
            f"{HNSW_EF_SEARCH}"
        )
    else:
        print(
            f"IVFFlat probes: "
            f"{IVFFLAT_PROBES}"
        )

    print(f"Loading model: {MODEL_NAME}")

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
            args.method,
        )

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

            exact_ids = None

            methods = [
                "exact",
                args.method,
            ]

            for method in methods:
                print(
                    f"Testing {method}..."
                )

                # Warm-up runs are excluded.
                for _ in range(
                    WARMUP_RUNS
                ):
                    run_search(
                        connection,
                        embedding,
                        method,
                    )

                latencies = []
                recalls = []

                for run_number in range(
                    1,
                    MEASURED_RUNS + 1,
                ):
                    (
                        latency_ms,
                        paper_ids,
                    ) = run_search(
                        connection,
                        embedding,
                        method,
                    )

                    if (
                        method == "exact"
                        and exact_ids is None
                    ):
                        exact_ids = paper_ids

                    if method == "exact":
                        recall = 1.0
                    else:
                        recall = (
                            calculate_recall(
                                exact_ids,
                                paper_ids,
                            )
                        )

                    latencies.append(
                        latency_ms
                    )
                    recalls.append(recall)

                    raw_results.append(
                        {
                            "query_number": (
                                query_number
                            ),
                            "query": query,
                            "method": method,
                            "run": run_number,
                            "latency_ms": (
                                latency_ms
                            ),
                            "recall_at_10": (
                                recall
                            ),
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

                mean_recall = float(
                    np.mean(recalls)
                )

                summary_results.append(
                    {
                        "query_number": (
                            query_number
                        ),
                        "query": query,
                        "method": method,
                        "mean_latency_ms": (
                            mean_latency
                        ),
                        "p95_latency_ms": (
                            p95_latency
                        ),
                        "recall_at_10": (
                            mean_recall
                        ),
                    }
                )

                print(
                    f"  Mean latency: "
                    f"{mean_latency:.3f} ms"
                )
                print(
                    f"  P95 latency:  "
                    f"{p95_latency:.3f} ms"
                )
                print(
                    f"  Recall@10:    "
                    f"{mean_recall:.1%}"
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
            f"{args.method}_raw.csv"
        )
    )

    summary_output = (
        results_directory
        / (
            f"benchmark_{args.count}_"
            f"{args.method}_summary.csv"
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

    overall = (
        summary_dataframe.groupby(
            "method"
        )
        .agg(
            mean_latency_ms=(
                "mean_latency_ms",
                "mean",
            ),
            mean_p95_latency_ms=(
                "p95_latency_ms",
                "mean",
            ),
            mean_recall_at_10=(
                "recall_at_10",
                "mean",
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        overall.to_string(
            index=False
        )
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
    print("Benchmark completed.")


if __name__ == "__main__":
    main()