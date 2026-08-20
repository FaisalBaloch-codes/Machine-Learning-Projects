"""
scrapers/combined_scrapers.py

Combined multi-source news scraper for ~8,000 article URLs.

Reads URLs from data/8000_urls.json, automatically detects which news
site each URL belongs to (based on hostname), scrapes the required
fields using ThreadPoolExecutor for parallel (multi-threaded) fetching,
and writes results incrementally to data/8000_articles.csv so that the
script can be safely stopped and resumed later.

Source detection and extraction are NOT limited to CNN and BBC. Any
domain found in the URLs is recognized and given a readable source
name automatically. CNN and BBC keep their own fine-tuned content
selectors (reused from the original bbc_scraper.py / cnn_scraper.py
reference files); every other site is scraped with a generic
extraction path (JSON-LD -> Open Graph -> meta tags -> common article
container selectors -> paragraph fallback), so adding new sites to the
input JSON requires no code changes.

Required output columns (exactly these, in this order):
    url, title, content, description, author, publish_date, source

Run with:
    python scrapers/combined_scrapers.py
"""

import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

URLS_JSON = "data/8000_urls.json"
OUTPUT_CSV = "data/8000_articles.csv"

# Only these columns are allowed in the final CSV.
CSV_FIELDS = [
    "url",
    "title",
    "content",
    "description",
    "author",
    "publish_date",
    "source",
]

# Number of worker threads used by ThreadPoolExecutor.
# Scraping is I/O-bound (waiting on network responses), so threads work
# well here. Start small (10) to avoid hammering CNN/BBC servers or
# triggering rate limiting; increase later only if it proves safe.
MAX_WORKERS = 10

REQUEST_TIMEOUT = 15  # seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Retry temporary errors (rate limiting / server errors) with a small
# exponential backoff: 2s, 4s, 8s, ...
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 2
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

# A single lock protects the CSV file and shared progress counters so
# that only one thread touches them at a time. Worker threads never
# write to the CSV directly - only the main thread does (see run_scraper).
csv_lock = threading.Lock()


# ============================================================
# STEP 1: LOAD URLS FROM JSON
# ============================================================

def load_urls(filepath=URLS_JSON):
    """
    Loads URLs from data/8000_urls.json.

    Supports two JSON shapes:
        ["https://...", "https://...", ...]
    or:
        [{"url": "https://..."}, {"url": "https://..."}, ...]

    Returns a de-duplicated list of URL strings, preserving first-seen
    order. Does not invent or modify URLs.
    """
    with open(filepath, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    raw_urls = []

    for item in data:
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            url = str(item.get("url", "")).strip()
        else:
            url = ""

        if url:
            raw_urls.append(url)
                                                                               
    # Remove duplicates while keeping the original order.
    seen = set()
    unique_urls = []
    for url in raw_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
            
    return unique_urls


# ============================================================
# STEP 2: SOURCE DETECTION (works for any number of news domains)
# ============================================================

# Readable names for sites we specifically fine-tune extraction for.
# Add more entries here any time a site earns its own custom selectors
# in extract_article_content() below.
KNOWN_SOURCE_NAMES = {
    "cnn.com": "CNN",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
}

# Common two-part country-code TLDs, needed so "bbc.co.uk" (not just
# "co.uk") is treated as the registrable domain. Extend this list if
# the URL set includes other country-specific news sites (e.g.
# "reuters.co.jp", "news.com.au").
TWO_PART_TLDS = {
    "co.uk", "co.in", "co.jp", "co.nz", "co.za",
    "com.au", "com.br", "com.mx", "com.sg",
}


def _registrable_domain(hostname):
    """
    Reduces a hostname like 'edition.cnn.com' or 'www.bbc.co.uk' down to
    its registrable domain ('cnn.com', 'bbc.co.uk'), so any subdomain of
    a known site is still recognized correctly.
    """
    parts = hostname.split(".")

    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_PART_TLDS:
        return ".".join(parts[-3:])

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return hostname

                            
def detect_source(url):
    """
    Determines which news site a URL belongs to, based on its hostname.
    Works regardless of subdomain (e.g. edition.cnn.com,
    subdomain.bbc.co.uk) and regardless of the order URLs appear in.

    Not limited to CNN/BBC: any domain is recognized and given a
    readable source name automatically. Sites in KNOWN_SOURCE_NAMES use
    their proper name (and their own fine-tuned extraction selectors);
    every other domain gets a name derived from the domain itself (e.g.
    'nytimes.com' -> 'Nytimes', 'reuters.com' -> 'Reuters') and is
    scraped using the generic extraction path.

    Only URLs with no parsable hostname at all fall back to UNKNOWN.
    """
    hostname = (urlparse(url).hostname or "").lower()

    if not hostname:
        return "UNKNOWN"

    domain = _registrable_domain(hostname)

    if domain in KNOWN_SOURCE_NAMES:
        return KNOWN_SOURCE_NAMES[domain]

    # Generic case: turn the domain's main label into a readable name.
    # e.g. "nytimes.com" -> "nytimes" -> "Nytimes"
    label = domain.split(".")[0]

    if not label:
        return "UNKNOWN"

    return label.capitalize()


# ============================================================
# STEP 3: FETCH PAGE HTML (WITH RETRY / BACKOFF)
# ============================================================

def fetch_article(url):
    """
    Downloads the HTML for one article URL.

    Returns (html, None) on success, or (None, error_message) on
    failure. Never raises - all request problems are caught here so a
    single bad URL can never crash the whole scraper.

    Retries temporary errors (429/500/502/503/504) with exponential
    backoff (2s, 4s, 8s, ...) up to MAX_RETRIES times. Other errors
    (timeouts, connection errors, permanent HTTP errors) are not
    retried endlessly - they are recorded as failed immediately.
    """
    attempt = 0

    while True:
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            return None, "Request timed out"
        except requests.exceptions.ConnectionError as error:
            return None, f"Connection error: {error}"
        except requests.exceptions.RequestException as error:
            return None, f"Request failed: {error}"

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
            wait_seconds = RETRY_BACKOFF_BASE_SECONDS ** (attempt + 1)
            time.sleep(wait_seconds)
            attempt += 1
            continue

        if response.status_code != 200:
            return None, f"HTTP error: status code {response.status_code}"

        return response.text, None


# ============================================================
# STEP 4: JSON-LD / METADATA HELPERS (shared by CNN + BBC)
# ============================================================

def extract_json_ld(soup):
    """
    Searches the page for JSON-LD structured data and returns the first
    Article/NewsArticle object found (or {} if none exists / parses).
    """
    for script_tag in soup.find_all("script", type="application/ld+json"):
        raw_text = (script_tag.string or script_tag.get_text() or "").strip()
        if not raw_text:
            continue

        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            continue

        candidates = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and "@graph" in data:
            candidates = data["@graph"]

        for item in candidates:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")
            type_values = item_type if isinstance(item_type, list) else [item_type]
            if any("Article" in str(t) for t in type_values):
                return item

    return {}


def _get_meta_content(soup, attr_name, attr_value):
    """Returns the 'content' attribute of a <meta> tag, or '' if missing."""
    tag = soup.find("meta", attrs={attr_name: attr_value})
    if tag and tag.get("content"):
        return tag.get("content").strip()
    return ""


def _get_author_from_jsonld(jsonld_author):
    """Normalizes JSON-LD 'author' (string, dict, or list) into a string."""
    if not jsonld_author:
        return ""
    if isinstance(jsonld_author, str):
        return jsonld_author
    if isinstance(jsonld_author, dict):
        return jsonld_author.get("name", "")
    if isinstance(jsonld_author, list):
        names = []
        for entry in jsonld_author:
            if isinstance(entry, dict):
                name = entry.get("name")
                if name:
                    names.append(name)
            elif isinstance(entry, str):
                names.append(entry)
        return ", ".join(names)
    return ""


def extract_metadata(soup):
    """
    Extracts title, description, author, and publish_date using the
    same fallback chain for both CNN and BBC:
        JSON-LD -> Open Graph meta tags -> standard meta tags -> <title>
    """
    jsonld = extract_json_ld(soup)

    title = (
        jsonld.get("headline")
        or _get_meta_content(soup, "property", "og:title")
        or _get_meta_content(soup, "name", "title")
        or (soup.title.string.strip() if soup.title and soup.title.string else "")
    )

    description = (
        jsonld.get("description")
        or _get_meta_content(soup, "property", "og:description")
        or _get_meta_content(soup, "name", "description")
    )

    author = (
        _get_author_from_jsonld(jsonld.get("author"))
        or _get_meta_content(soup, "name", "author")
    )

    publish_date = (
        jsonld.get("datePublished")
        or _get_meta_content(soup, "property", "article:published_time")
        or _get_meta_content(soup, "name", "date")
    )

    return {
        "title": title or "",
        "description": description or "",
        "author": author or "",
        "publish_date": publish_date or "",
    }


# ============================================================
# STEP 5: ARTICLE CONTENT EXTRACTION (CNN + BBC specific)
# ============================================================

def _extract_paragraphs(soup, candidate_selectors):
    """
    Shared helper: strips out nav/header/footer/script/style/aside, then
    tries each selector in order looking for article paragraphs, falling
    back to every <p> tag on the page as a last resort. Returns cleaned,
    de-duplicated article text.
    """
    for unwanted in soup.select("nav, header, footer, script, style, aside"):
        unwanted.decompose()

    paragraphs = []
    for selector in candidate_selectors:
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all("p")
            if paragraphs:
                break

    if not paragraphs:
        paragraphs = soup.find_all("p")

    text_pieces = []
    seen = set()
    for paragraph in paragraphs:
        text = paragraph.get_text(" ", strip=True)
        if len(text) > 20 and text not in seen:
            seen.add(text)
            text_pieces.append(text)

    return "\n".join(text_pieces)


CNN_SELECTORS = [
    "div.article__content",
    "div.article__main",
    "section.article__content",
    "div[itemprop='articleBody']",
    "article",
]

BBC_SELECTORS = [
    "[itemprop='articleBody']",
    "article",
    "main article",
    "main",
]

# Fallback selectors for any site that isn't CNN or BBC. These cover the
# container patterns most news publishers use in practice. Tried in
# order; if none of them yield paragraphs, _extract_paragraphs() still
# falls back to every <p> tag on the page as a last resort.
GENERIC_SELECTORS = [
    "[itemprop='articleBody']",
    "div.article-body",
    "div.article__body",
    "div.story-body",
    "div.entry-content",
    "div.post-content",
    "article",
    "main article",
    "main",
]


def extract_cnn_article(soup):
    """Extracts CNN article body text, preferring CNN's known containers."""
    return _extract_paragraphs(soup, CNN_SELECTORS + GENERIC_SELECTORS)


def extract_bbc_article(soup):
    """Extracts BBC article body text, preferring BBC's known containers."""
    return _extract_paragraphs(soup, BBC_SELECTORS + GENERIC_SELECTORS)


def extract_generic_article(soup):
    """
    Extracts article body text for any site other than CNN/BBC, using
    common publisher container patterns with a paragraph-tag fallback.
    This is what makes the scraper work for new sites without any code
    changes - it doesn't need site-specific selectors to find text.
    """
    return _extract_paragraphs(soup, GENERIC_SELECTORS)


def extract_article_content(soup, source):
    """
    Dispatches to the right content extractor based on source. CNN and
    BBC get their own fine-tuned selectors (tried first, with the
    generic selectors as a fallback); every other recognized source
    uses the generic extractor directly.
    """
    if source == "CNN":
        return extract_cnn_article(soup)
    if source == "BBC":
        return extract_bbc_article(soup)
    if source == "UNKNOWN":
        return ""
    return extract_generic_article(soup)


# ============================================================
# STEP 6: SCRAPE ONE ARTICLE (runs inside a worker thread)
# ============================================================

def scrape_article(url):
    """
    Scrapes a single article and returns a result dict matching
    CSV_FIELDS. This function runs inside worker threads - it only
    fetches and parses; it never touches the CSV file directly.
    """
    source = detect_source(url)

    row = {
        "url": url,
        "title": "",
        "content": "",
        "description": "",
        "author": "",
        "publish_date": "",
        "source": source,
    }

    if source == "UNKNOWN":
        # Only happens if the URL has no parsable hostname at all - a
        # genuinely malformed URL, not just "a site other than CNN/BBC".
        print(f"UNKNOWN source, skipping scrape - {url}")
        return row, False, "Unknown source (no parsable hostname)"

    html, error = fetch_article(url)
    if error:
        return row, False, error

    try:
        soup = BeautifulSoup(html, "html.parser")
        metadata = extract_metadata(soup)
        content = extract_article_content(soup, source)

        row["title"] = metadata["title"]
        row["description"] = metadata["description"]
        row["author"] = metadata["author"]
        row["publish_date"] = metadata["publish_date"]
        row["content"] = content

        return row, True, None

    except Exception as error:
        return row, False, f"Parsing error: {error}"


# ============================================================
# STEP 7: RESUME SUPPORT - LOAD ALREADY-PROCESSED URLS
# ============================================================

def load_processed_urls(filename=OUTPUT_CSV):
    """
    Reads the existing output CSV (if any) and returns the set of URLs
    that were SUCCESSFULLY scraped (i.e. have real content, not just a
    placeholder row), so they can be skipped on this run.

    A row only counts as "done" if title or content is non-empty. Rows
    that only have url + source (failed scrapes - network loss, timeouts,
    HTTP errors, etc.) are treated as NOT processed, so they get
    retried instead of being skipped forever.
    """
    if not os.path.exists(filename):
        return set()

    processed = set()
    with open(filename, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            url = row.get("url", "").strip()
            title = (row.get("title") or "").strip()
            content = (row.get("content") or "").strip()
            if url and (title or content):
                processed.add(url)

    return processed


def strip_failed_rows(filename=OUTPUT_CSV):
    """
    Rewrites the CSV keeping only rows that were successfully scraped
    (title or content non-empty). Failed placeholder rows (url + source
    only) are dropped from the file here so that when those URLs are
    retried, append_to_csv() adds a single fresh row instead of creating
    a duplicate entry for the same URL.

    Safe to call even if the file doesn't exist yet, or has no failed
    rows to remove.
    """
    if not os.path.exists(filename):
        return 0

    kept_rows = []
    removed_count = 0

    with open(filename, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            title = (row.get("title") or "").strip()
            content = (row.get("content") or "").strip()
            if title or content:
                kept_rows.append(row)
            else:
                removed_count += 1

    if removed_count == 0:
        return 0

    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)

    return removed_count


# ============================================================
# STEP 8: INCREMENTAL CSV WRITING (main thread only)
# ============================================================

def ensure_csv_header(filename=OUTPUT_CSV):
    """
    Creates the output CSV (and its data/ directory) with the header
    row if it doesn't already exist. If it exists, leaves it untouched
    so previously scraped rows are preserved.
    """
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_to_csv(row, filename=OUTPUT_CSV):
    """
    Appends a single completed result to the CSV. Only ever called from
    the main thread (never from worker threads) so writes can't
    interleave or corrupt the file. The lock is extra protection in
    case this function is ever called from more than one place.
    """
    with csv_lock:
        with open(filename, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writerow(row)


# ============================================================
# STEP 9: RUN THE SCRAPER (ThreadPoolExecutor + as_completed)
# ============================================================

def run_scraper():
    """
    Orchestrates the full scrape:
        1. Load + de-duplicate URLs from JSON.
        2. Skip URLs already present in the output CSV (resume support).
        3. Submit remaining URLs to a ThreadPoolExecutor (worker threads
           only fetch/parse - they never write to the CSV).
        4. As each worker finishes, the MAIN THREAD receives the result
           via as_completed() and immediately appends it to the CSV.
           This is what makes saving incremental: if the script stops
           halfway through, everything already written stays on disk.
    """
    start_time = time.perf_counter()

    all_urls = load_urls()
    total_unique = len(all_urls)

    ensure_csv_header()

    # Remove any leftover failed placeholder rows (url + source only,
    # no title/content) from a previous interrupted run, so those URLs
    # get a clean retry instead of being skipped or duplicated.
    removed_count = strip_failed_rows()

    already_processed = load_processed_urls()

    urls_to_scrape = [u for u in all_urls if u not in already_processed]
    skipped_count = total_unique - len(urls_to_scrape)

    print("=" * 50)
    print(f"Total URLs in JSON (unique): {total_unique}")
    print(f"Previously failed rows cleared for retry: {removed_count}")
    print(f"Already successfully processed (skipped): {skipped_count}")
    print(f"Remaining to scrape (new + retries): {len(urls_to_scrape)}")
    print("=" * 50)

    if not urls_to_scrape:
        print("Nothing left to scrape. All URLs already processed.")
        return

    success_count = 0
    failed_count = 0
    processed_count = 0
    total_to_process = len(urls_to_scrape)

    # ThreadPoolExecutor manages a fixed pool of MAX_WORKERS threads -
    # it does NOT create one thread per URL. Each worker thread runs
    # scrape_article() for one URL at a time and simply returns its
    # result; only this main thread writes to the CSV or updates
    # shared progress counters.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(scrape_article, url): url
            for url in urls_to_scrape
        }

        # as_completed() yields futures as soon as each one finishes,
        # in whatever order they happen to complete - NOT the original
        # submission order. This lets us save results immediately
        # instead of waiting for the slowest URL to catch up.
        for future in as_completed(futures):
            url = futures[future]
            processed_count += 1

            try:
                row, success, error_message = future.result()
            except Exception as error:
                # Safety net: a worker should never raise, but just in
                # case, record it as a failed row instead of crashing.
                row = {
                    "url": url,
                    "title": "",
                    "content": "",
                    "description": "",
                    "author": "",
                    "publish_date": "",
                    "source": detect_source(url),
                }
                success = False
                error_message = f"Unexpected worker error: {error}"

            # Main thread: append immediately, one row at a time.
            append_to_csv(row)

            if success:
                success_count += 1
                print(f"[{processed_count}/{total_to_process}] {row['source']} SUCCESS - {url}")
            else:
                failed_count += 1
                print(f"[{processed_count}/{total_to_process}] {row['source']} FAILED - {error_message} - {url}")

            # Periodic progress summary every 1000 articles.
            if processed_count % 1000 == 0:
                elapsed = time.perf_counter() - start_time
                minutes, seconds = divmod(int(elapsed), 60)
                print("-" * 50)
                print(f"Processed: {processed_count}/{total_to_process}")
                print(f"Successful: {success_count}")
                print(f"Failed: {failed_count}")
                print(f"Elapsed: {minutes}m {seconds}s")
                print("-" * 50)

    elapsed_time = time.perf_counter() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)

    print()
    print("=" * 50)
    print("SCRAPING COMPLETE")
    print("=" * 50)
    print(f"Total URLs: {total_unique}")
    print(f"Unique URLs: {total_unique}")
    print(f"Skipped already processed: {skipped_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Elapsed time: {minutes}m {seconds}s")
    print(f"Output: {OUTPUT_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():
    run_scraper()


if __name__ == "__main__":
    main()