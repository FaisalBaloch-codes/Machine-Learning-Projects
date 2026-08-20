"""
scrapers/cnn_scraper.py

Phase 2, CNN only: reads the 250 CNN URLs from data/cnn_urls.csv, fetches
each page, pulls out raw structured fields (from JSON-LD, Open Graph tags,
standard meta tags, and the article body), and saves everything to
data/cnn_raw_articles.csv.

This step does NOT clean or normalize the data - it saves what's found,
as found, so cleaning can be done properly later with Pandas.
"""

import csv
import os
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup


# --- Configuration ---

CNN_URLS_CSV = "data/cnn_urls.csv"
OUTPUT_CSV = "data/cnn_raw_articles.csv"

REQUEST_TIMEOUT = 10          # seconds
DELAY_BETWEEN_REQUESTS = 1.5  # seconds, to avoid hammering the server

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_RETRIES = 2            # extra attempts after the first, only for 503/429
RETRY_DELAY_SECONDS = 3    # pause before retrying a temporary server error

# IMPORTANT: set this to True while you're testing on a single URL.
# Only switch it to False once you've verified the single-article test
# output looks correct (title/description/author/date/content all make sense).
TEST_MODE = False


# --- robots.txt check ---

_robot_parsers = {}  # cache one RobotFileParser per domain so we don't refetch robots.txt every time


def is_allowed_by_robots(url):
    """
    Checks the target site's robots.txt to confirm we're allowed to fetch
    this URL before requesting it. If robots.txt itself can't be reached
    (network issue, timeout, etc.), we don't block the run on it - only
    an actual robots.txt rule should block a URL.
    """
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain not in _robot_parsers:
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{domain}/robots.txt")

        read_successfully = True
        try:
            rp.read()
        except Exception:
            read_successfully = False

        _robot_parsers[domain] = (rp, read_successfully)

    rp, read_successfully = _robot_parsers[domain]

    if not read_successfully:
        # Couldn't check robots.txt at all - don't block on it
        return True

    try:
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


# --- Step 1: load URLs to scrape ---

def load_cnn_urls(filepath=CNN_URLS_CSV):
    """
    Reads data/cnn_urls.csv and returns a list of URL strings from the
    'url' column.
    """
    urls = []
    with open(filepath, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            url = row.get("url", "").strip()
            if url:
                urls.append(url)
    return urls


# --- Step 2: fetch the page ---

def fetch_article(url):
    """
    Downloads the HTML for a single article URL.

    Returns a tuple: (html_text, error_message)
    - On success: (html_string, None)
    - On failure: (None, "description of what went wrong")

    Never raises an exception - all request problems are caught and
    turned into an error message instead.
    """
    if not is_allowed_by_robots(url):
        return None, "Blocked by robots.txt"

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

        if response.status_code in (503, 429) and attempt < MAX_RETRIES:
            attempt += 1
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        if response.status_code != 200:
            return None, f"HTTP error: status code {response.status_code}"

        return response.text, None


# --- Step 3: pull structured metadata out of the page ---

def _get_json_ld_data(soup):
    """
    Looks through every <script type="application/ld+json"> block on the
    page and returns the first one that looks like a NewsArticle (or
    Article). CNN pages commonly embed this. Returns {} if none found or
    if the JSON can't be parsed.
    """
    for script_tag in soup.find_all("script", type="application/ld+json"):
        raw_text = script_tag.string or script_tag.get_text() or ""
        raw_text = raw_text.strip()
        if not raw_text:
            continue
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            continue

        # Some pages wrap multiple JSON-LD objects in a list, or use @graph
        candidates = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and "@graph" in data:
            candidates = data["@graph"]

        for item in candidates:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")
            # @type can be a string or a list of strings
            type_values = item_type if isinstance(item_type, list) else [item_type]
            if any("Article" in str(t) for t in type_values):
                return item

    return {}


def _get_meta_content(soup, attr_name, attr_value):
    """Helper: returns the 'content' attribute of a <meta> tag, or '' if missing."""
    tag = soup.find("meta", attrs={attr_name: attr_value})
    if tag and tag.get("content"):
        return tag.get("content").strip()
    return ""


def _get_author_from_jsonld(jsonld_author):
    """
    JSON-LD 'author' can be a dict, a list of dicts, or a plain string.
    This normalizes it down to a readable string without discarding info.
    """
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


def _guess_section_from_url(url):
    """
    Fallback only: CNN article URLs often look like
    /2026/08/10/health/example-story/index.html - the segment after the
    date is usually the section. Used only if no metadata provides one.
    """
    path_parts = [p for p in urlparse(url).path.split("/") if p]
    for part in path_parts:
        if not part.isdigit() and part.lower() != "index.html":
            return part
    return ""


def extract_metadata(soup, url):
    """
    Pulls together title, description, author, dates, category/section,
    keywords, image, canonical URL, article type, and language from
    JSON-LD (preferred, most structured) and falls back to Open Graph /
    standard meta tags when JSON-LD doesn't have a value.

    Returns a dictionary. Any field that truly can't be found is left as
    an empty string - fields are never dropped or skipped.
    """
    jsonld = _get_json_ld_data(soup)

    title = (
        jsonld.get("headline")
        or _get_meta_content(soup, "property", "og:title")
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
    )

    last_updated = (
        jsonld.get("dateModified")
        or _get_meta_content(soup, "property", "article:modified_time")
    )

    section = (
        jsonld.get("articleSection")
        or _get_meta_content(soup, "property", "article:section")
        or _guess_section_from_url(url)
    )

    # keywords: JSON-LD 'keywords' can be a string or a list
    raw_keywords = jsonld.get("keywords") or _get_meta_content(soup, "name", "keywords")
    if isinstance(raw_keywords, list):
        keywords = ", ".join(str(k) for k in raw_keywords)
    else:
        keywords = raw_keywords or ""

    image_url = ""
    jsonld_image = jsonld.get("image")
    if isinstance(jsonld_image, dict):
        image_url = jsonld_image.get("url", "")
    elif isinstance(jsonld_image, list) and jsonld_image:
        first = jsonld_image[0]
        image_url = first.get("url", "") if isinstance(first, dict) else str(first)
    elif isinstance(jsonld_image, str):
        image_url = jsonld_image
    if not image_url:
        image_url = _get_meta_content(soup, "property", "og:image")

    canonical_tag = soup.find("link", rel="canonical")
    canonical_url = canonical_tag.get("href", "").strip() if canonical_tag else ""

    article_type = jsonld.get("@type", "") or _get_meta_content(soup, "property", "og:type")
    if isinstance(article_type, list):
        article_type = ", ".join(str(t) for t in article_type)

    html_tag = soup.find("html")
    language = html_tag.get("lang", "") if html_tag else ""

    # CNN doesn't always label "category" separately from "section" -
    # record both; if category isn't found anywhere, fall back to section
    category = _get_meta_content(soup, "property", "article:section") or section

    return {
        "title": title or "",
        "description": description or "",
        "author": author or "",
        "publish_date": publish_date or "",
        "last_updated": last_updated or "",
        "category": category or "",
        "section": section or "",
        "keywords": keywords or "",
        "image_url": image_url or "",
        "canonical_url": canonical_url or "",
        "article_type": article_type or "",
        "language": language or "",
    }


# --- Step 4: extract the actual article body text ---

def extract_article_content(soup):
    """
    Attempts to pull just the article's body paragraphs, skipping nav
    menus, ads, related-story boxes, and footers.

    Strategy: prefer paragraphs inside an <article> tag or CNN's known
    article body containers; fall back to all <p> tags on the page only
    if nothing more specific is found.
    """
    # Remove obviously non-article elements before searching, so their
    # text can't leak into whatever container we pick
    for unwanted in soup.select("nav, header, footer, script, style, aside"):
        unwanted.decompose()

    # CNN's article body has used class names like these over time;
    # try the most specific containers first
    candidate_selectors = [
        "div.article__content",
        "div.article__main",
        "section.article__content",
        "div[itemprop='articleBody']",
        "article",
    ]

    paragraphs = []
    for selector in candidate_selectors:
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all("p")
            if paragraphs:
                break

    # Last resort: grab all <p> tags on the page
    if not paragraphs:
        paragraphs = soup.find_all("p")

    # Filter out very short fragments (often captions, bylines, or UI text)
    text_pieces = [p.get_text(strip=True) for p in paragraphs]
    text_pieces = [t for t in text_pieces if len(t) > 20]

    content = "\n".join(text_pieces)
    return content


# --- Step 5: put it all together for one article ---

def scrape_cnn_article(url):
    """
    Fetches and parses a single CNN article URL, returning one row
    (a dictionary) matching the output CSV's columns. Never raises -
    any failure is captured in scrape_status / error_message instead.
    """
    row = {
        "url": url,
        "source": "CNN",
        "title": "",
        "description": "",
        "author": "",
        "publish_date": "",
        "last_updated": "",
        "category": "",
        "section": "",
        "keywords": "",
        "content": "",
        "image_url": "",
        "canonical_url": "",
        "article_type": "",
        "language": "",
        "word_count": 0,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "scrape_status": "failed",
        "error_message": "",
    }

    html, error = fetch_article(url)
    if error:
        row["error_message"] = error
        return row

    try:
        soup = BeautifulSoup(html, "html.parser")
        metadata = extract_metadata(soup, url)
        content = extract_article_content(soup)

        row.update(metadata)
        row["content"] = content
        row["word_count"] = len(content.split()) if content else 0
        row["scrape_status"] = "success"
    except Exception as error:
        row["error_message"] = f"Parsing error: {error}"
        row["scrape_status"] = "failed"

    return row


# --- Step 6: save all rows to CSV ---

RAW_ARTICLE_FIELDS = [
    "url", "source", "title", "description", "author", "publish_date",
    "last_updated", "category", "section", "keywords", "content",
    "image_url", "canonical_url", "article_type", "language",
    "word_count", "scraped_at", "scrape_status", "error_message",
]


def save_raw_articles(rows, filename=OUTPUT_CSV):
    """Writes the list of article-row dictionaries to a CSV file."""
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RAW_ARTICLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows written to CSV: {len(rows)}")
    print(f"Output file: {filename}")


# --- Main ---

def run_single_url_test():
    """Fetches and prints just ONE article so results can be verified by eye."""
    urls = load_cnn_urls()
    if not urls:
        print("No URLs found in data/cnn_urls.csv - nothing to test.")
        return

    test_url = urls[0]
    print(f"=== Testing scraper on 1 URL ===\n{test_url}\n")

    row = scrape_cnn_article(test_url)

    for field, value in row.items():
        if field == "content":
            preview = (value[:500] + "...") if len(value) > 500 else value
            print(f"{field}: {preview}")
        else:
            print(f"{field}: {value}")

    print("\nCheck the fields above. If title/description/author/date/content")
    print("all look correct, set TEST_MODE = False near the top of this file")
    print("and run the script again to process all 250 URLs.")


def run_full_scrape():
    """Scrapes every URL in data/cnn_urls.csv and saves the results."""
    urls = load_cnn_urls()
    total = len(urls)

    if total == 0:
        print("No URLs found in data/cnn_urls.csv - nothing to scrape.")
        return

    print(f"=== Scraping {total} CNN articles ===")

    rows = []
    success_count = 0
    failed_count = 0

    for index, url in enumerate(urls, start=1):
        row = scrape_cnn_article(url)
        rows.append(row)

        if row["scrape_status"] == "success":
            success_count += 1
        else:
            failed_count += 1

        print(f"Scraping CNN: {index}/{total}")

        # Be polite - pause between requests (skip the delay after the last one)
        if index < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    save_raw_articles(rows)

    print("\n=== CNN SCRAPING COMPLETE ===")
    print(f"Total URLs processed: {total}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Output file: {OUTPUT_CSV}")


if __name__ == "__main__":
    if TEST_MODE:
        run_single_url_test()
    else:
        run_full_scrape()