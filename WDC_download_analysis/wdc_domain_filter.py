#!/usr/bin/env python3
"""
WDC JSON-LD Domain Filter (v2 — rate-limit-safe)
==================================================
Downloads WDC 2024-12 JSON-LD N-Quad files ONE AT A TIME with generous
pauses between downloads to avoid 429s from the WDC server.

Key differences from v1:
  - Sequential downloads (1 worker) — no parallelism
  - Long pause between files (default 30s, configurable)
  - Aggressive backoff on 429: starts at 120s, doubles each retry
  - Respects Retry-After header if present
  - Failed files go to a retry queue processed at the end with even longer waits

Usage:
    python wdc_domain_filter.py [--pause 30] [--output results.nq]

Resume-safe: tracks progress in checkpoint.json, safe to Ctrl+C and restart.
"""

import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────────────────

FILE_LIST_URL = "https://webdatacommons.org/structureddata/2024-12/files/html-embedded-jsonld.list"

TARGET_DOMAINS = [
    "en.wikipedia.org",
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "fullfact.org",
    "africacheck.org",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "aljazeera.com",
    "france24.com",
    "dw.com",
    "theguardian.com",
    "nytimes.com",
    "thetimes.com",
    "lemonde.fr",
    "cnn.com",
]

# Build fast-lookup patterns
DOMAIN_PATTERNS = []
for d in TARGET_DOMAINS:
    DOMAIN_PATTERNS.append(f"://{d}/")
    DOMAIN_PATTERNS.append(f"://{d}>")
    DOMAIN_PATTERNS.append(f".{d}/")
    DOMAIN_PATTERNS.append(f".{d}>")

CHECKPOINT_FILE = "checkpoint.json"
TEMP_DIR = "wdc_temp"

# ── Helpers ────────────────────────────────────────────────────────────────


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "match_count": 0, "files_processed": 0}


def save_checkpoint(state):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT_FILE)


def fetch_file_list(session: requests.Session) -> list[str]:
    """Download the .list file and return list of URLs."""
    print(f"Fetching file list from {FILE_LIST_URL} ...")
    for attempt in range(3):
        try:
            resp = session.get(FILE_LIST_URL, timeout=60)
            resp.raise_for_status()
            urls = [line.strip() for line in resp.text.strip().splitlines() if line.strip()]
            print(f"  → {len(urls)} files in list")
            return urls
        except requests.RequestException as e:
            print(f"  ⚠ Failed to fetch file list (attempt {attempt+1}): {e}")
            time.sleep(10)
    raise RuntimeError("Could not fetch file list after 3 attempts")


def line_matches(line: str) -> bool:
    """Check if an N-Quad line originates from one of our target domains."""
    for pattern in DOMAIN_PATTERNS:
        if pattern in line:
            return True
    return False


def download_and_filter(
    url: str, session: requests.Session, output_path: str, max_retries: int = 6
) -> tuple[bool, int, int]:
    """
    Download one .gz file, filter, append matches, delete.
    Returns (success, matches, total_lines).

    On 429: respects Retry-After header, or backs off starting at 120s.
    """
    filename = url.split("/")[-1]
    local_path = os.path.join(TEMP_DIR, filename)

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=180, stream=True)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after) + random.uniform(5, 15)
                else:
                    wait = (120 * (2 ** attempt)) + random.uniform(0, 30)
                print(f"  ⏳ 429 on {filename} (attempt {attempt+1}/{max_retries}), "
                      f"waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue

            resp.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            # Filter
            matches = 0
            total_lines = 0
            matched_lines = []

            with gzip.open(local_path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    total_lines += 1
                    if line_matches(line):
                        matched_lines.append(line)
                        matches += 1

            if matched_lines:
                with open(output_path, "a", encoding="utf-8") as out:
                    out.writelines(matched_lines)

            return (True, matches, total_lines)

        except (requests.RequestException, IOError, OSError) as e:
            wait = (60 * (2 ** attempt)) + random.uniform(0, 15)
            print(f"  ⚠ Error on {filename} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"    Waiting {wait:.0f}s before retry ...")
                time.sleep(wait)

        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    return (False, 0, 0)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Filter WDC JSON-LD by domain (v2)")
    parser.add_argument(
        "--pause",
        type=int,
        default=30,
        help="Seconds to pause between successful downloads (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wdc_filtered_domains.nq",
        help="Output file for matching quads (default: wdc_filtered_domains.nq)",
    )
    args = parser.parse_args()

    os.makedirs(TEMP_DIR, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "WDC-Research-Filter/1.0 (academic research; "
            "contact: your-email@example.com)"
        }
    )

    all_urls = fetch_file_list(session)

    state = load_checkpoint()
    completed_set = set(state["completed"])
    failed_set = set(state.get("failed", []))
    remaining = [u for u in all_urls if u not in completed_set and u not in failed_set]

    total_matches = state["match_count"]
    files_done = state["files_processed"]
    total_files = len(all_urls)

    print(f"\nResume state: {files_done}/{total_files} files done, "
          f"{total_matches:,} matches so far")
    print(f"Remaining: {len(remaining)} files  |  Previously failed: {len(failed_set)}")
    print(f"Pause between files: {args.pause}s")
    print(f"Output: {args.output}")
    print(f"Target domains: {len(TARGET_DOMAINS)}")
    print("─" * 65)

    # ── Process remaining files sequentially ──
    for i, url in enumerate(remaining):
        filename = url.split("/")[-1]
        success, matches, total_lines = download_and_filter(
            url, session, args.output
        )

        files_done += 1

        if success:
            total_matches += matches
            state["completed"].append(url)
            state["match_count"] = total_matches
            state["files_processed"] = files_done

            pct = (files_done / total_files) * 100
            status = f"✓ {matches:,} matches" if matches else "  0 matches"
            print(
                f"[{files_done:>5}/{total_files}] ({pct:5.1f}%) "
                f"{filename}  {status}  "
                f"({total_lines:,} lines) | "
                f"cumulative: {total_matches:,}"
            )
        else:
            state["failed"].append(url)
            state["files_processed"] = files_done
            print(f"[{files_done:>5}/{total_files}] ✗ FAILED {filename}")

        # Checkpoint every file
        save_checkpoint(state)

        # Pause between files
        if i < len(remaining) - 1:
            pause = args.pause + random.uniform(0, 10)
            time.sleep(pause)

    # ── Retry previously failed files with extra-long pauses ──
    failed_urls = list(failed_set) + [
        u for u in state.get("failed", []) if u not in failed_set
    ]
    if failed_urls:
        print(f"\n{'─' * 65}")
        print(f"Retrying {len(failed_urls)} previously failed files "
              f"with 90s pauses ...")
        print("─" * 65)

        state["failed"] = []
        save_checkpoint(state)

        for i, url in enumerate(failed_urls):
            filename = url.split("/")[-1]

            if i > 0:
                time.sleep(90 + random.uniform(0, 30))

            success, matches, total_lines = download_and_filter(
                url, session, args.output, max_retries=4
            )

            if success:
                total_matches += matches
                state["completed"].append(url)
                state["match_count"] = total_matches
                status = f"✓ {matches:,} matches" if matches else "  0 matches"
                print(f"  [retry] {filename}  {status}")
            else:
                state["failed"].append(url)
                print(f"  [retry] ✗ FAILED again: {filename}")

            save_checkpoint(state)

    # ── Summary ──
    final_failed = len(state.get("failed", []))
    print(f"\n{'─' * 65}")
    print(f"Done! {total_matches:,} matching quads saved to {args.output}")
    print(f"Files processed: {len(state['completed'])}/{total_files}")
    if final_failed:
        print(f"Files still failed: {final_failed} (listed in {CHECKPOINT_FILE})")
    print("─" * 65)


if __name__ == "__main__":
    main()
