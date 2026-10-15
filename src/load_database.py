import argparse
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        choices=[500, 5000, 10000, 30000],
    )
    args = parser.parse_args()

    data_file = Path(
        f"data/processed/arxiv_{args.count}.csv"
    )
    embedding_file = Path(
        f"data/processed/"
        f"arxiv_{args.count}_embeddings.npy"
    )

    papers = pd.read_csv(data_file)
    embeddings = np.load(embedding_file)

    if len(papers) != args.count:
        raise ValueError(
            f"Expected {args.count} papers, "
            f"but found {len(papers)}."
        )

    if embeddings.shape != (args.count, 384):
        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    records = []

    for paper, embedding in zip(
        papers.itertuples(index=False),
        embeddings,
    ):
        records.append(
            (
                paper.arxiv_id,
                paper.title,
                paper.abstract,
                paper.category,
                date.fromisoformat(
                    paper.publication_date
                ),
                embedding,
            )
        )

    print(
        f"Replacing database contents with "
        f"{args.count} papers..."
    )
    start_time = perf_counter()

    with psycopg.connect(
        host="localhost",
        port=5433,
        dbname="cs779_vector_db",
        user="postgres",
        password="cs779password",
    ) as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                "TRUNCATE TABLE papers RESTART IDENTITY;"
            )

            cursor.executemany(
                """
                INSERT INTO papers (
                    arxiv_id,
                    title,
                    abstract,
                    category,
                    publication_date,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                records,
            )

            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    MIN(vector_dims(embedding)),
                    MAX(vector_dims(embedding))
                FROM papers
                WHERE embedding IS NOT NULL;
                """
            )

            row_count, min_dimensions, max_dimensions = (
                cursor.fetchone()
            )

        connection.commit()

    elapsed_seconds = perf_counter() - start_time

    print(f"Rows with embeddings: {row_count}")
    print(
        f"Embedding dimensions: "
        f"{min_dimensions} to {max_dimensions}"
    )
    print(
        f"Database loading time: "
        f"{elapsed_seconds:.2f} seconds"
    )
    print("Database loading completed.")


if __name__ == "__main__":
    main()