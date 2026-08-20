"""
scrapers/bbc_scraper.py

Phase 2, BBC only: reads the 250 BBC URLs from data/bbc_urls.csv,
fetches each page, extracts the required article fields, and saves
the raw results to data/bbc_raw_articles.csv.

Required scraped fields:
- title
- description
- author
- publish_date
- content

This step does NOT clean or normalize the data.
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


# ============================================================
# CONFIGURATION
# ============================================================

BBC_URLS_CSV = "data/bbc_urls.csv"
OUTPUT_CSV = "data/bbc_raw_articles.csv"

REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3

# Keep True for testing one article first.
# Change to False after checking the test result.
TEST_MODE = False


# ============================================================
# ROBOTS.TXT CHECK
# ============================================================

_robot_parsers = {}


def is_allowed_by_robots(url):
    """
    Checks BBC robots.txt before requesting the page.
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
        return True

    try:
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


# ============================================================
# STEP 1: LOAD BBC URLS
# ============================================================

def load_bbc_urls(filepath=BBC_URLS_CSV):
    """
    Reads BBC URLs from data/bbc_urls.csv.
   
    The CSV must contain a column named 'url'.
    """

    urls = []

    with open(
        filepath,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            url = row.get("url", "").strip()

            if url:
                urls.append(url)

    return urls


# ============================================================
# STEP 2: FETCH BBC ARTICLE
# ============================================================

def fetch_article(url):
    """
    Downloads HTML for one BBC article.

    Returns:

        (html, None)

    on success.

    Returns:

        (None, error_message)

    on failure.
    """

    if not is_allowed_by_robots(url):
        return None, "Blocked by robots.txt"

    attempt = 0

    while True:

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )

        except requests.exceptions.Timeout:

            return None, "Request timed out"

        except requests.exceptions.ConnectionError as error:

            return None, f"Connection error: {error}"

        except requests.exceptions.RequestException as error:

            return None, f"Request failed: {error}"

        # Retry temporary server/rate-limit errors
        if response.status_code in (503, 429):

            if attempt < MAX_RETRIES:

                attempt += 1

                time.sleep(RETRY_DELAY_SECONDS)

                continue

        if response.status_code != 200:

            return None, (
                f"HTTP error: status code "
                f"{response.status_code}"
            )

        return response.text, None


# ============================================================
# STEP 3: FIND JSON-LD ARTICLE DATA
# ============================================================

def _get_json_ld_data(soup):
    """
    Searches the page for JSON-LD structured data.

    BBC pages commonly contain article information
    inside JSON-LD.

    Returns the first Article/NewsArticle object found.
    """

    for script_tag in soup.find_all(
        "script",
        type="application/ld+json"
    ):

        raw_text = (
            script_tag.string
            or script_tag.get_text()
            or ""
        ).strip()

        if not raw_text:
            continue

        try:

            data = json.loads(raw_text)

        except (json.JSONDecodeError, TypeError):

            continue

        # Handle lists
        candidates = (
            data
            if isinstance(data, list)
            else [data]
        )

        # Handle @graph
        if isinstance(data, dict) and "@graph" in data:

            candidates = data["@graph"]

        for item in candidates:

            if not isinstance(item, dict):
                continue

            item_type = item.get("@type", "")

            type_values = (
                item_type
                if isinstance(item_type, list)
                else [item_type]
            )

            if any(
                "Article" in str(article_type)
                for article_type in type_values
            ):

                return item

    return {}


# ============================================================
# STEP 4: META TAG HELPER
# ============================================================

def _get_meta_content(
    soup,
    attr_name,
    attr_value
):
    """
    Returns the content value of a meta tag.
    """

    tag = soup.find(
        "meta",
        attrs={
            attr_name: attr_value
        }
    )

    if tag and tag.get("content"):

        return tag.get("content").strip()

    return ""


# ============================================================
# STEP 5: AUTHOR EXTRACTION
# ============================================================

def _get_author_from_jsonld(jsonld_author):
    """
    Handles different JSON-LD author formats.
    """

    if not jsonld_author:
        return ""

    if isinstance(jsonld_author, str):
        return jsonld_author

    if isinstance(jsonld_author, dict):

        return jsonld_author.get(
            "name",
            ""
        )

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


# ============================================================
# STEP 6: EXTRACT BBC METADATA
# ============================================================

def extract_metadata(soup):
    """
    Extracts only the required BBC fields:

    - title
    - description
    - author
    - publish_date
    """

    jsonld = _get_json_ld_data(soup)

    # -------------------------
    # TITLE
    # -------------------------

    title = (
        jsonld.get("headline")
        or _get_meta_content(
            soup,
            "property",
            "og:title"
        )
        or _get_meta_content(
            soup,
            "name",
            "title"
        )
        or (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else ""
        )
    )

    # -------------------------
    # DESCRIPTION
    # -------------------------

    description = (
        jsonld.get("description")
        or _get_meta_content(
            soup,
            "property",
            "og:description"
        )
        or _get_meta_content(
            soup,
            "name",
            "description"
        )
    )

    # -------------------------
    # AUTHOR
    # -------------------------

    author = (
        _get_author_from_jsonld(
            jsonld.get("author")
        )
        or _get_meta_content(
            soup,
            "name",
            "author"
        )
    )

    # -------------------------
    # PUBLISH DATE
    # -------------------------

    publish_date = (
        jsonld.get("datePublished")
        or _get_meta_content(
            soup,
            "property",
            "article:published_time"
        )
        or _get_meta_content(
            soup,
            "name",
            "date"
        )
    )

    return {
        "title": title or "",
        "description": description or "",
        "author": author or "",
        "publish_date": publish_date or "",
    }


# ============================================================
# STEP 7: EXTRACT BBC ARTICLE CONTENT
# ============================================================

def extract_article_content(soup):
    """
    Attempts to extract only the main BBC article text.

    Strategy:

    1. Look for BBC article containers.
    2. Look for <article>.
    3. Look for elements using articleBody.
    4. Fall back to paragraph tags.

    Navigation, scripts, headers, footers and sidebars
    are removed first.
    """

    # Remove obvious non-article elements
    for unwanted in soup.select(
        "nav, header, footer, script, style, aside"
    ):

        unwanted.decompose()

    # BBC pages can change their class names over time.
    candidate_selectors = [

        # Schema/article body
        "[itemprop='articleBody']",

        # Standard article element
        "article",

        # Common BBC article containers
        "main article",

        "main",
    ]

    paragraphs = []

    for selector in candidate_selectors:

        container = soup.select_one(selector)

        if container:

            paragraphs = container.find_all("p")

            if paragraphs:
                break

    # Last resort
    if not paragraphs:

        paragraphs = soup.find_all("p")

    # Extract paragraph text
    text_pieces = []

    for paragraph in paragraphs:

        text = paragraph.get_text(
            " ",
            strip=True
        )

        # Ignore extremely short UI fragments
        if len(text) > 20:

            text_pieces.append(text)

    # Remove exact duplicate paragraphs
    unique_pieces = []

    seen = set()

    for text in text_pieces:

        if text not in seen:

            seen.add(text)
            unique_pieces.append(text)

    content = "\n".join(unique_pieces)

    return content


# ============================================================
# STEP 8: SCRAPE ONE BBC ARTICLE
# ============================================================

def scrape_bbc_article(url):
    """
    Scrapes one BBC article and returns one CSV row.
    """

    row = {

        "url": url,

        "title": "",

        "description": "",

        "author": "",

        "publish_date": "",

        "content": "",

        "scrape_status": "failed",

        "error_message": "",
    }

    # Fetch page
    html, error = fetch_article(url)

    if error:

        row["error_message"] = error

        return row

    try:

        # Parse HTML
        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # Extract metadata
        metadata = extract_metadata(soup)

        # Extract article content
        content = extract_article_content(soup)

        # Update row
        row.update(metadata)

        row["content"] = content

        # Mark successful
        row["scrape_status"] = "success"

    except Exception as error:

        row["error_message"] = (
            f"Parsing error: {error}"
        )

        row["scrape_status"] = "failed"

    return row


# ============================================================
# STEP 9: SAVE DATA TO CSV
# ============================================================

RAW_ARTICLE_FIELDS = [

    "url",

    "title",

    "description",

    "author",

    "publish_date",

    "content",

    "scrape_status",

    "error_message",
]


def save_raw_articles(
    rows,
    filename=OUTPUT_CSV
):
    """
    Saves scraped BBC article data to CSV.
    """

    output_dir = os.path.dirname(filename)

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=RAW_ARTICLE_FIELDS
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Rows written to CSV: {len(rows)}"
    )

    print(
        f"Output file: {filename}"
    )


# ============================================================
# STEP 10: TEST ONE ARTICLE
# ============================================================

def run_single_url_test():
    """
    Tests the scraper using only the first BBC URL.
    """

    urls = load_bbc_urls()

    if not urls:

        print(
            "No URLs found in "
            "data/bbc_urls.csv - nothing to test."
        )

        return

    test_url = urls[0]

    print(
        "=== Testing BBC scraper on 1 URL ==="
    )

    print(test_url)
    print()

    row = scrape_bbc_article(test_url)

    for field, value in row.items():

        if field == "content":

            preview = (
                value[:1000] + "..."
                if len(value) > 1000
                else value
            )

            print(
                f"{field}: {preview}"
            )

        else:

            print(
                f"{field}: {value}"
            )

    print()
    print(
        "Check title, description, author, "
        "publish_date and content."
    )

    print(
        "If they look correct, change "
        "TEST_MODE = False and run again."
    )


# ============================================================
# STEP 11: SCRAPE ALL 250 BBC ARTICLES
# ============================================================

def run_full_scrape():
    """
    Scrapes every BBC URL in data/bbc_urls.csv.
    """

    urls = load_bbc_urls()

    total = len(urls)

    if total == 0:

        print(
            "No URLs found in "
            "data/bbc_urls.csv - nothing to scrape."
        )

        return

    print(
        f"=== Scraping {total} BBC articles ==="
    )

    rows = []

    success_count = 0
    failed_count = 0

    for index, url in enumerate(
        urls,
        start=1
    ):

        row = scrape_bbc_article(url)

        rows.append(row)

        if row["scrape_status"] == "success":

            success_count += 1

        else:

            failed_count += 1

        print(
            f"Scraping BBC: "
            f"{index}/{total}"
        )

        # Delay between requests
        if index < total:

            time.sleep(
                DELAY_BETWEEN_REQUESTS
            )

    # Save CSV
    save_raw_articles(rows)

    # Final summary
    print()
    print(
        "=== BBC SCRAPING COMPLETE ==="
    )

    print(
        f"Total URLs processed: {total}"
    )

    print(
        f"Successful: {success_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    print(
        f"Output file: {OUTPUT_CSV}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if TEST_MODE:

        run_single_url_test()

    else:

        run_full_scrape()