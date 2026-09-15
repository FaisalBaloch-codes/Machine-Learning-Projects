"""
Fetch weekly dark-pool (ATS) volume data from FINRA's public OTC Transparency
API. No API key required.

Endpoint reference:
    POST https://api.finra.org/data/group/otcMarket/name/weeklySummary
    Body: {"compareFilters": [...], "dateRangeFilters": [...], "limit": N,
           "offset": N}

This is genuine, working code against FINRA's documented schema. It requires
outbound network access to api.finra.org. If that host is not reachable from
the current environment (e.g. a sandboxed egress allowlist), this script will
raise a clear connection error rather than silently failing -- run it from a
machine/notebook with normal internet access.

Output: data/raw/finra_weekly_ats_raw.csv
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TICKERS, START_DATE, END_DATE, FINRA_BASE_URL, FINRA_TIER, DATA_RAW

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def fetch_symbol(symbol: str, start_date: str, end_date: str, session: requests.Session,
                  max_retries: int = 3, timeout: int = 20) -> list:
    """Fetch ALL weekly ATS records for one symbol within a date range.

    FINRA's weekly ATS summary has one row PER REPORTING VENUE per week (many
    dark pools report separately for the same symbol), and synchronous
    requests are capped at 5000 rows. We paginate with offset/limit until a
    short page tells us we've reached the end.
    """
    all_records = []
    offset = 0
    limit = 5000

    while True:
        payload = {
            "compareFilters": [
                {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": symbol},
            ],
            "dateRangeFilters": [
                {
                    "fieldName": "weekStartDate",
                    "startDate": start_date,
                    "endDate": end_date,
                }
            ],
            "limit": limit,
            "offset": offset,
        }

        page = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.post(FINRA_BASE_URL, data=json.dumps(payload), headers=HEADERS, timeout=timeout)
                resp.raise_for_status()
                page = resp.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise
                print(f"  [{symbol}] attempt {attempt} failed ({e}); retrying...")
                time.sleep(1.5 * attempt)

        if not page:
            break
        all_records.extend(page)
        if len(page) < limit:
            break  # last page
        offset += limit
        time.sleep(0.2)

    return all_records


def main():
    session = requests.Session()
    all_rows = []
    failures = []

    print(f"Fetching FINRA weekly ATS data for {len(TICKERS)} tickers, "
          f"{START_DATE} to {END_DATE} ...")

    for i, symbol in enumerate(TICKERS, 1):
        print(f"[{i}/{len(TICKERS)}] {symbol}")
        try:
            records = fetch_symbol(symbol, START_DATE, END_DATE, session)
            for r in records:
                r["query_symbol"] = symbol
            all_rows.extend(records)
        except Exception as e:
            print(f"  FAILED: {symbol} -> {e}")
            failures.append(symbol)
        time.sleep(0.3)  # be polite to the public API

    if not all_rows:
        raise RuntimeError(
            "No data returned from FINRA API for any ticker. This almost always means "
            "the current environment cannot reach api.finra.org (check network/egress "
            "settings) rather than a problem with the request format."
        )

    df = pd.DataFrame(all_rows)
    out_path = DATA_RAW / "finra_weekly_ats_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows -> {out_path}")
    if failures:
        print(f"Failed tickers ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
