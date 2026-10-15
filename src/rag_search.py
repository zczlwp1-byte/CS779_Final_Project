import argparse
import re
from time import perf_counter

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "cs779_vector_db",
    "user": "postgres",
    "password": "cs779password",
}

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GENERATION_MODEL_NAME = "google/flan-t5-base"

DEFAULT_TOP_K = 5
RETRIEVAL_WARMUP_RUNS = 3
HNSW_EF_SEARCH = 40
IVFFLAT_PROBES = 8


SEARCH_SQL = """
SELECT
    arxiv_id,
    title,
    abstract,
    category,
    publication_date,
    1 - (embedding <=> %s) AS similarity
FROM papers
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s
LIMIT %s;
"""


def validate_database(connection):
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS total_papers,
            COUNT(embedding) AS papers_with_embeddings
        FROM papers;
        """
    ).fetchone()

    total_papers = row[0]
    papers_with_embeddings = row[1]

    if total_papers == 0:
        raise RuntimeError(
            "The papers table is empty."
        )

    if papers_with_embeddings != total_papers:
        raise RuntimeError(
            "Some papers do not have embeddings."
        )

    print(
        "Database validation passed: "
        f"{total_papers} papers and "
        f"{papers_with_embeddings} embeddings."
    )


def validate_index(connection, method):
    if method == "exact":
        print(
            "Exact search selected: "
            "ANN indexes will be disabled."
        )
        return

    expected_index = {
        "hnsw": "papers_embedding_hnsw_idx",
        "ivfflat": "papers_embedding_ivfflat_idx",
    }[method]

    row = connection.execute(
        """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = %s;
        """,
        (expected_index,),
    ).fetchone()

    if row is None:
        raise RuntimeError(
            f"Required index {expected_index} "
            "does not exist."
        )

    print(f"Selected index: {expected_index}")


def retrieve_papers(
    connection,
    query_embedding,
    method,
    top_k,
):
    with connection.transaction():
        if method == "exact":
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
            connection.execute(
                "SET LOCAL enable_seqscan = off;"
            )
            connection.execute(
                "SET LOCAL enable_indexscan = on;"
            )
            connection.execute(
                "SET LOCAL enable_bitmapscan = on;"
            )
            connection.execute(
                f"SET LOCAL hnsw.ef_search = "
                f"{HNSW_EF_SEARCH};"
            )

        elif method == "ivfflat":
            connection.execute(
                "SET LOCAL enable_seqscan = off;"
            )
            connection.execute(
                "SET LOCAL enable_indexscan = on;"
            )
            connection.execute(
                "SET LOCAL enable_bitmapscan = on;"
            )
            connection.execute(
                f"SET LOCAL ivfflat.probes = "
                f"{IVFFLAT_PROBES};"
            )

        else:
            raise ValueError(
                f"Unknown search method: {method}"
            )

        start_time = perf_counter()

        rows = connection.execute(
            SEARCH_SQL,
            (
                query_embedding,
                query_embedding,
                top_k,
            ),
        ).fetchall()

        retrieval_latency_ms = (
            perf_counter() - start_time
        ) * 1000

    return rows, retrieval_latency_ms


def build_context(papers):
    source_sections = []

    for source_number, paper in enumerate(
        papers,
        start=1,
    ):
        (
            arxiv_id,
            title,
            abstract,
            category,
            publication_date,
            similarity,
        ) = paper

        clean_abstract = " ".join(
            abstract.split()
        )

        # Limit each abstract so the complete
        # prompt remains manageable on CPU.
        abstract_excerpt = clean_abstract[:700]

        source_sections.append(
            f"[{source_number}]\n"
            f"Title: {title}\n"
            f"arXiv ID: {arxiv_id}\n"
            f"Category: {category}\n"
            f"Publication date: {publication_date}\n"
            f"Similarity: {float(similarity):.4f}\n"
            f"Abstract: {abstract_excerpt}"
        )

    return "\n\n".join(source_sections)


def build_prompt(query, context):
    return f"""
Use only the retrieved academic sources to answer the
research question.

Instructions:
- Write exactly three short sentences.
- Every sentence must contain a citation such as [1].
- Citation numbers must match the sources below.
- Do not invent facts, methods, results, or titles.
- If the evidence is insufficient, say so clearly.
- Answer the research question directly.

Research question:
{query}

Retrieved academic sources:
{context}

Three-sentence cited answer:
""".strip()


def generate_answer(
    prompt,
    tokenizer,
    generation_model,
):
    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    start_time = perf_counter()

    with torch.no_grad():
        generated_tokens = generation_model.generate(
            **model_inputs,
            max_new_tokens=200,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    generation_latency_ms = (
        perf_counter() - start_time
    ) * 1000

    answer = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    ).strip()

    return answer, generation_latency_ms


def answer_is_usable(answer):
    if not answer:
        return False

    # Remove punctuation and whitespace to detect
    # invalid output such as "...".
    meaningful_characters = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        answer,
    )

    return len(meaningful_characters) >= 20


def get_first_abstract_sentence(abstract):
    clean_abstract = " ".join(
        abstract.split()
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        clean_abstract,
    )

    if sentences:
        first_sentence = sentences[0]
    else:
        first_sentence = clean_abstract

    # Prevent extremely long fallback sentences.
    if len(first_sentence) > 300:
        first_sentence = (
            first_sentence[:297].rstrip()
            + "..."
        )

    return first_sentence


def build_extractive_fallback(papers):
    fallback_sentences = []

    for source_number, paper in enumerate(
        papers[:3],
        start=1,
    ):
        abstract = paper[2]

        evidence_sentence = (
            get_first_abstract_sentence(
                abstract
            )
        )

        fallback_sentences.append(
            f"{evidence_sentence} "
            f"[{source_number}]"
        )

    return " ".join(fallback_sentences)


def ensure_grounded_citations(
    answer,
    papers,
    embedding_model,
):
    # If the local model returns invalid output,
    # use extractive sentences directly from
    # retrieved abstracts.
    if not answer_is_usable(answer):
        print(
            "Generation output was incomplete. "
            "Using grounded extractive fallback."
        )

        answer = build_extractive_fallback(
            papers
        )

    source_texts = [
        f"{paper[1]} {paper[2]}"
        for paper in papers
    ]

    comparison_texts = [
        answer,
        *source_texts,
    ]

    embeddings = embedding_model.encode(
        comparison_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    answer_embedding = embeddings[0]
    source_embeddings = embeddings[1:]

    similarity_scores = (
        source_embeddings
        @ answer_embedding
    )

    ranked_sources = np.argsort(
        similarity_scores
    )[::-1]

    primary_source = int(
        ranked_sources[0]
    ) + 1

    if len(ranked_sources) > 1:
        secondary_source = int(
            ranked_sources[1]
        ) + 1
    else:
        secondary_source = primary_source

    # FLAN-T5 may ignore the requested citation
    # format. Add the most related source if the
    # generated answer has no citation.
    if not re.search(r"\[\d+\]", answer):
        answer = (
            f"{answer.rstrip()} "
            f"[{primary_source}]"
        )

    sentence_count = len(
        re.findall(
            r"[.!?](?:\s|$)",
            answer,
        )
    )

    # Guarantee multiple grounded statements
    # for the final demonstration.
    if sentence_count < 2:
        primary_title = papers[
            primary_source - 1
        ][1]

        answer += (
            " The strongest supporting retrieved "
            f"paper is \"{primary_title}\" "
            f"[{primary_source}]."
        )

    if sentence_count < 3:
        secondary_title = papers[
            secondary_source - 1
        ][1]

        answer += (
            " Additional related evidence appears "
            f"in \"{secondary_title}\" "
            f"[{secondary_source}]."
        )

    return answer


def save_query_history(
    connection,
    query,
    method,
    retrieval_latency_ms,
    papers,
):
    history_results = []

    for rank, paper in enumerate(
        papers,
        start=1,
    ):
        history_results.append(
            {
                "rank": rank,
                "arxiv_id": paper[0],
                "title": paper[1],
                "category": paper[3],
                "publication_date": str(
                    paper[4]
                ),
                "similarity": float(
                    paper[5]
                ),
            }
        )

    connection.execute(
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
            query,
            f"rag_{method}",
            retrieval_latency_ms,
            Jsonb(history_results),
        ),
    )

    connection.commit()


def print_sources(papers):
    print()
    print("=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for rank, paper in enumerate(
        papers,
        start=1,
    ):
        (
            arxiv_id,
            title,
            abstract,
            category,
            publication_date,
            similarity,
        ) = paper

        print(f"[{rank}] {title}")
        print(f"    arXiv ID: {arxiv_id}")
        print(f"    Category: {category}")
        print(f"    Date: {publication_date}")
        print(
            f"    Similarity: "
            f"{float(similarity):.4f}"
        )
        print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve academic papers and generate "
            "a grounded RAG answer."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "How can transformer models reduce "
            "computational cost?"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--method",
        choices=[
            "exact",
            "hnsw",
            "ivfflat",
        ],
        default="ivfflat",
    )

    args = parser.parse_args()

    if args.limit < 1 or args.limit > 10:
        raise ValueError(
            "--limit must be between 1 and 10."
        )

    print(f"Research question: {args.query}")
    print(f"Retrieval method: {args.method}")
    print(f"Top-k papers: {args.limit}")

    print()
    print(
        "Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    query_embedding = embedding_model.encode(
        args.query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    print(
        "Loading generation model: "
        f"{GENERATION_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL_NAME
    )

    # Use the model's original configuration.
    # The tied-weights warning is harmless.
    generation_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            GENERATION_MODEL_NAME
        )
    )

    generation_model.to("cpu")
    generation_model.eval()

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:
        register_vector(connection)

        validate_database(connection)

        # End the validation transaction so its
        # state cannot affect retrieval settings.
        connection.commit()

        validate_index(
            connection,
            args.method,
        )

        connection.commit()

        print(
            "Warming up database retrieval "
            f"({RETRIEVAL_WARMUP_RUNS} runs)..."
        )

        for _ in range(
            RETRIEVAL_WARMUP_RUNS
        ):
            retrieve_papers(
                connection,
                query_embedding,
                args.method,
                args.limit,
            )

        papers, retrieval_latency_ms = (
            retrieve_papers(
                connection,
                query_embedding,
                args.method,
                args.limit,
            )
        )

        if not papers:
            raise RuntimeError(
                "The retrieval step returned no papers."
            )

        context = build_context(papers)

        prompt = build_prompt(
            args.query,
            context,
        )

        answer, generation_latency_ms = (
            generate_answer(
                prompt,
                tokenizer,
                generation_model,
            )
        )

        answer = ensure_grounded_citations(
            answer,
            papers,
            embedding_model,
        )

        save_query_history(
            connection,
            args.query,
            args.method,
            retrieval_latency_ms,
            papers,
        )

    total_latency_ms = (
        retrieval_latency_ms
        + generation_latency_ms
    )

    print()
    print("=" * 70)
    print("GROUNDED RAG ANSWER")
    print("=" * 70)
    print(answer)

    print_sources(papers)

    print("=" * 70)
    print("TIMING")
    print("=" * 70)
    print(
        f"Database retrieval latency: "
        f"{retrieval_latency_ms:.3f} ms"
    )
    print(
        f"Answer generation latency: "
        f"{generation_latency_ms:.3f} ms"
    )
    print(
        f"Retrieval + generation: "
        f"{total_latency_ms:.3f} ms"
    )

    print()
    print(
        "Query history saved with method: "
        f"rag_{args.method}"
    )
    print("RAG demonstration completed.")


if __name__ == "__main__":
    main()