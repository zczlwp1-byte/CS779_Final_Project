import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        choices=[500, 5000, 10000, 30000],
    )
    args = parser.parse_args()

    input_file = Path(
        f"data/processed/arxiv_{args.count}.csv"
    )
    output_file = Path(
        f"data/processed/"
        f"arxiv_{args.count}_embeddings.npy"
    )

    papers = pd.read_csv(input_file)

    if len(papers) != args.count:
        raise ValueError(
            f"Expected {args.count} papers, "
            f"but found {len(papers)}."
        )

    texts = (
        papers["title"].fillna("")
        + ". "
        + papers["abstract"].fillna("")
    ).tolist()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print(
        f"Generating embeddings for "
        f"{len(texts)} papers..."
    )
    start_time = perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed_seconds = perf_counter() - start_time

    if embeddings.shape != (len(papers), 384):
        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.save(output_file, embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {output_file}")
    print(
        f"Generation time: "
        f"{elapsed_seconds:.2f} seconds"
    )
    print("Embedding generation completed.")


if __name__ == "__main__":
    main()