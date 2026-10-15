import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import feedparser
import requests


API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 500
REQUEST_DELAY_SECONDS = 5
MAX_RETRIES = 8

SEARCH_QUERY = (
    "cat:cs.AI OR cat:cs.CL OR cat:cs.IR "
    "OR cat:cs.LG OR cat:cs.DB"
)

FIELDNAMES = [
    "arxiv_id",
    "title",
    "abstract",
    "category",
    "publication_date",
]


def clean_text(text):
    return " ".join(text.split())


def clean_arxiv_id(entry_id):
    arxiv_id = entry_id.replace(
        "https://arxiv.org/abs/", ""
    ).replace(
        "http://arxiv.org/abs/", ""
    )

    return re.sub(r"v\d+$", "", arxiv_id)


def build_daily_query(current_day):
    day_text = current_day.strftime("%Y%m%d")

    return (
        f"({SEARCH_QUERY}) AND "
        f"submittedDate:["
        f"{day_text}0000 TO {day_text}2359]"
    )


def download_page(
    current_day,
    page_start,
    page_size,
):
    params = {
        "search_query": build_daily_query(
            current_day
        ),
        "start": page_start,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": (
            "CS779-Vector-Search-"
            "Student-Project/1.0"
        )
    }

    retry_waits = [
        15,
        30,
        60,
        90,
        120,
        180,
        240,
    ]

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = requests.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=300,
            )

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:
                print(
                    f"API returned "
                    f"{response.status_code}. "
                    f"Attempt "
                    f"{attempt}/{MAX_RETRIES}."
                )

                if attempt == MAX_RETRIES:
                    break

                wait_time = retry_waits[
                    attempt - 1
                ]

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            feed = feedparser.parse(
                response.content
            )

            if feed.bozo:
                print(
                    "Unable to parse response. "
                    f"Attempt "
                    f"{attempt}/{MAX_RETRIES}."
                )

                if attempt == MAX_RETRIES:
                    break

                wait_time = retry_waits[
                    attempt - 1
                ]

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            return feed.entries

        except requests.RequestException as error:
            print(f"Request failed: {error}")

            print(
                f"Attempt "
                f"{attempt}/{MAX_RETRIES}."
            )

            if attempt == MAX_RETRIES:
                break

            wait_time = retry_waits[
                attempt - 1
            ]

            print(
                f"Retrying in "
                f"{wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Unable to download the current day. "
        "Progress has been saved. Run the "
        "same command again to resume."
    )


def load_partial_records(partial_file):
    records_by_id = {}

    if not partial_file.exists():
        return records_by_id

    with partial_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for record in reader:
            records_by_id[
                record["arxiv_id"]
            ] = record

    return records_by_id


def save_records(file_path, records):
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(records)


def determine_start_day(records_by_id):
    if not records_by_id:
        return date.today()

    publication_dates = []

    for record in records_by_id.values():
        try:
            publication_dates.append(
                date.fromisoformat(
                    record["publication_date"]
                )
            )
        except ValueError:
            continue

    if not publication_dates:
        return date.today()

    return min(publication_dates)


def load_daily_checkpoint(
    checkpoint_file,
    default_day,
):
    if not checkpoint_file.exists():
        return default_day, 0

    try:
        with checkpoint_file.open(
            "r",
            encoding="utf-8",
        ) as checkpoint:
            checkpoint_data = json.load(
                checkpoint
            )

        # Ignore the older global/monthly checkpoint.
        if (
            checkpoint_data.get("download_mode")
            != "daily"
        ):
            return default_day, 0

        current_day = date.fromisoformat(
            checkpoint_data["current_day"]
        )

        day_page_start = int(
            checkpoint_data.get(
                "day_page_start",
                0,
            )
        )

        return current_day, day_page_start

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
    ):
        return default_day, 0


def save_checkpoint(
    checkpoint_file,
    current_day,
    day_page_start,
    unique_records,
):
    checkpoint_data = {
        "download_mode": "daily",
        "current_day": current_day.isoformat(),
        "day_page_start": day_page_start,
        "unique_records": unique_records,
    }

    with checkpoint_file.open(
        "w",
        encoding="utf-8",
    ) as checkpoint:
        json.dump(
            checkpoint_data,
            checkpoint,
            indent=2,
        )


def entry_to_record(entry):
    primary_category = entry.get(
        "arxiv_primary_category",
        {},
    ).get(
        "term",
        "",
    )

    arxiv_id = clean_arxiv_id(entry.id)

    record = {
        "arxiv_id": arxiv_id,
        "title": clean_text(entry.title),
        "abstract": clean_text(entry.summary),
        "category": primary_category,
        "publication_date": entry.published[:10],
    }

    return arxiv_id, record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=500,
        choices=[
            500,
            5000,
            10000,
            30000,
        ],
    )

    args = parser.parse_args()

    output_file = Path(
        f"data/processed/"
        f"arxiv_{args.count}.csv"
    )

    partial_file = Path(
        f"data/processed/"
        f"arxiv_{args.count}.partial.csv"
    )

    checkpoint_file = Path(
        f"data/processed/"
        f"arxiv_{args.count}.checkpoint.json"
    )

    records_by_id = load_partial_records(
        partial_file
    )

    default_day = determine_start_day(
        records_by_id
    )

    current_day, day_page_start = (
        load_daily_checkpoint(
            checkpoint_file,
            default_day,
        )
    )

    if records_by_id:
        print(
            f"Found saved progress: "
            f"{len(records_by_id)} "
            f"unique papers."
        )

        print(
            "Switching to daily date windows "
            "to avoid the API start=10000 error."
        )

        print(
            f"Resuming with day: "
            f"{current_day.isoformat()}"
        )
    else:
        print(
            f"Downloading {args.count} "
            f"arXiv papers..."
        )

        print(
            f"Starting with day: "
            f"{current_day.isoformat()}"
        )

    while len(records_by_id) < args.count:
        print()
        print(
            f"Day {current_day.isoformat()}: "
            f"downloading page starting at "
            f"{day_page_start}..."
        )

        entries = download_page(
            current_day,
            day_page_start,
            PAGE_SIZE,
        )

        if not entries:
            print(
                f"No more records on "
                f"{current_day.isoformat()}."
            )

            current_day -= timedelta(days=1)
            day_page_start = 0

            save_checkpoint(
                checkpoint_file,
                current_day,
                day_page_start,
                len(records_by_id),
            )

            continue

        added_count = 0

        for entry in entries:
            arxiv_id, record = entry_to_record(
                entry
            )

            if arxiv_id not in records_by_id:
                records_by_id[
                    arxiv_id
                ] = record

                added_count += 1

            if len(records_by_id) >= args.count:
                break

        day_page_start += len(entries)

        records = list(
            records_by_id.values()
        )

        save_records(
            partial_file,
            records,
        )

        if len(entries) < PAGE_SIZE:
            print(
                f"Finished day "
                f"{current_day.isoformat()}."
            )

            current_day -= timedelta(days=1)
            day_page_start = 0

        save_checkpoint(
            checkpoint_file,
            current_day,
            day_page_start,
            len(records_by_id),
        )

        print(
            f"Added {added_count} new papers."
        )

        print(
            f"Checkpoint saved: "
            f"{len(records_by_id)} "
            f"unique papers."
        )

        if len(records_by_id) < args.count:
            print(
                f"Waiting "
                f"{REQUEST_DELAY_SECONDS} "
                f"seconds before next request..."
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    records = list(
        records_by_id.values()
    )

    records.sort(
        key=lambda record: (
            record["publication_date"],
            record["arxiv_id"],
        ),
        reverse=True,
    )

    records = records[:args.count]

    save_records(
        output_file,
        records,
    )

    if partial_file.exists():
        partial_file.unlink()

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    categories = Counter(
        record["category"]
        for record in records
    )

    print()
    print(
        f"Saved {len(records)} papers "
        f"to {output_file}"
    )

    print("Most common primary categories:")

    for category, category_count in (
        categories.most_common(15)
    ):
        print(
            f"  {category}: "
            f"{category_count}"
        )

    print("Data preparation completed.")


if __name__ == "__main__":
    main()