import feedparser
from urllib.parse import urlparse
from datetime import date, timedelta
from urllib.parse import quote_plus
from googlenewsdecoder import gnewsdecoder
import csv
import os



# --- Feed lists ---
# A single RSS feed only returns a limited number of recent entries
# (you saw ~30 from the main BBC feed). To realistically get closer to
# 250 unique URLs per source, we read several *real* topic feeds from
# each publisher instead of inventing fake data.

CNN_RSS_FEEDS = [
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "http://rss.cnn.com/rss/cnn_world.rss",
    "http://rss.cnn.com/rss/cnn_us.rss",
    "http://rss.cnn.com/rss/cnn_allpolitics.rss",
    "http://rss.cnn.com/rss/money_latest.rss",
    "http://rss.cnn.com/rss/cnn_tech.rss",
    "http://rss.cnn.com/rss/cnn_health.rss",
    "http://rss.cnn.com/rss/cnn_showbiz.rss",
]

BBC_RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/uk/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "http://feeds.bbci.co.uk/news/health/rss.xml",
    "http://feeds.bbci.co.uk/news/education/rss.xml",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
]

TARGET_COUNT = 250  # how many unique URLs we want per source

def build_google_news_rss_url(query, after_date=None, before_date=None,
                               hl="en-US", gl="US", ceid="US:en"):
    """
    Builds a Google News RSS *search* URL that feedparser can read, using
    the 'before:' / 'after:' date operators to search a specific date
    range instead of only the latest headlines.
    """
    full_query = query
    if after_date:
        full_query += f" after:{after_date}"
    if before_date:
        full_query += f" before:{before_date}"

    encoded_query = quote_plus(full_query)
    return (
        f"https://news.google.com/rss/search?q={encoded_query}"
        f"&hl={hl}&gl={gl}&ceid={ceid}"
    )


def generate_date_ranges(window_days=7, max_ranges=20):
    """
    Builds a list of (after_date, before_date) string pairs, stepping
    backwards from today in `window_days`-sized chunks.
    """
    ranges = []
    today = date.today()
    for i in range(max_ranges):
        before = today - timedelta(days=i * window_days)
        after = before - timedelta(days=window_days)
        ranges.append((after.isoformat(), before.isoformat()))
    return ranges


def decode_google_news_link(google_news_url):
    """
    Safely decodes a Google News redirect link into the original
    publisher URL. Returns None on any failure so a bad link never
    crashes the loop.
    """
    try:
        result = gnewsdecoder(google_news_url, interval=1)
    except Exception as error:
        print(f"  Decoder error for {google_news_url}: {error}")
        return None

    if not result or not result.get("status"):
        return None

    return result.get("decoded_url")


def collect_cnn_urls_from_google_news(cnn_urls, target_count=250,
                                       window_days=7, max_ranges=20):
    """
    Tops up an EXISTING set of CNN URLs using Google News as a secondary
    discovery source, searching multiple before/after date ranges, until
    `target_count` unique CNN URLs are reached.

    `cnn_urls` should be your existing 200-URL set/collection. This
    function adds to it in place and also returns it.

    Assumes is_cnn_url() and is_google_news_url() already exist in this
    file (from your existing CNN/BBC domain-checking code).
    """
    date_ranges = generate_date_ranges(window_days, max_ranges)

    for after_date, before_date in date_ranges:
        if len(cnn_urls) >= target_count:
            break

        feed_url = build_google_news_rss_url(
            "site:cnn.com", after_date=after_date, before_date=before_date
        )

        try:
            entries = read_feed(feed_url, verbose=False)
        except Exception as error:
            print(f"Could not read Google News feed for {after_date} to {before_date}: {error}")
            continue

        for entry in entries:
            if len(cnn_urls) >= target_count:
                break

            link = entry.get("link", "")
            if not link:
                continue

            if is_google_news_url(link):
                article_url = decode_google_news_link(link)
                if not article_url:
                    continue
            elif is_cnn_url(link):
                article_url = link
            else:
                continue

            if not is_cnn_url(article_url):
                continue

            if article_url in cnn_urls:
                continue

            cnn_urls.add(article_url)
            print(f"CNN URLs collected: {len(cnn_urls)}/{target_count}")

    if len(cnn_urls) < target_count:
        print(f"Note: Google News search only found {len(cnn_urls)} total unique CNN URLs "
              f"(target was {target_count}). This is an honest limit of what the search "
              f"turned up - try a larger window_days or max_ranges. No fake URLs were added.")

    return cnn_urls




def read_feed(feed_url, verbose=True):
    """
    Reads an RSS feed from the given URL, optionally prints basic info
    (title, link, published date) for each entry, and returns the
    collected entries as a list of dictionaries.

    `verbose=False` silences per-entry printing, which is useful when this
    function is called many times in a row (once per feed in the list)
    and printing every entry would flood the terminal.
    """
    # Parse the feed. feedparser handles downloading and parsing for us.
    feed = feedparser.parse(feed_url)

    # feedparser sets 'bozo' to True if something went wrong while parsing.
    if feed.bozo:
        if verbose:
            print(f"Warning: Failed to parse feed properly. Reason: {feed.bozo_exception}")
        return []

    if not feed.entries:
        if verbose:
            print("No entries found in this feed.")
        return []

    if verbose:
        print(f"Found {len(feed.entries)} entries.\n")

    collected_entries = []

    for index, entry in enumerate(feed.entries, start=1):
        # Use .get() so the code doesn't crash if a field is missing
        title = entry.get("title", "No title available")
        link = entry.get("link", "No link available")
        published = entry.get("published", "No published date available")

        if verbose:
            print(f"Entry {index}")
            print(f"Title     : {title}")
            print(f"Link      : {link}")
            print(f"Published : {published}")
            print("-" * 40)

        collected_entries.append({
            "title": title,
            "link": link,
            "published": published
        })

    return collected_entries


# --- Simple domain-checking helpers ---
# These just look at the URL text itself; no network calls happen here.
                            
def get_domain(url):
    """Returns the lowercase domain (netloc) part of a URL, or '' on error."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_cnn_url(url):
    """True if the URL's domain is cnn.com or a cnn.com subdomain."""
    domain = get_domain(url)
    return domain == "cnn.com" or domain.endswith(".cnn.com")


def is_bbc_url(url):
    """True if the URL's domain is a bbc.co.uk or bbc.com (sub)domain."""
    domain = get_domain(url)
    return domain.endswith("bbc.co.uk") or domain.endswith("bbc.com")

# --- Article-content path filters (rejects video/audio/live pages) ---

CNN_REJECT_PATH_PARTS = (
    "/videos/",
    "/video/",
    "/audio/",
    "/podcasts/",
    "/podcast/",
    "/live-news/",
)


def is_cnn_article_url(url):
    """
    True only if the URL looks like a normal CNN text news article
    (e.g. https://www.cnn.com/2023/07/09/health/example/index.html),
    not a video, audio/podcast, or live-news page.
    """
    path = urlparse(url).path.lower()

    if not path or path == "/":
        return False

    if any(part in path for part in CNN_REJECT_PATH_PARTS):
        return False

    return True


BBC_REJECT_PATH_PARTS = (
    "/sounds/",
    "/av/",
    "/live/",
    "/video/",
    "/videos/",
    "/programmes/",
)


def is_bbc_article_url(url):
    """
    True only if the URL looks like a normal BBC News text article
    (e.g. https://www.bbc.co.uk/news/articles/... or the older
    https://www.bbc.co.uk/news/uk-12345678 style), not BBC Sounds,
    a video page, or a live page.
    """
    path = urlparse(url).path.lower()

    # BBC News text articles always live under /news/...
    if not path.startswith("/news"):
        return False

    if any(part in path for part in BBC_REJECT_PATH_PARTS):
        return False

    return True


def is_valid_article_url(url):
    """
    True only if the URL both belongs to CNN/BBC AND passes the
    article-content path rules above. Use this wherever you currently
    only check is_cnn_url()/is_bbc_url() before accepting a link.
    """
    if is_cnn_url(url):
        return is_cnn_article_url(url)
    if is_bbc_url(url):
        return is_bbc_article_url(url)
    return False

def is_google_news_url(url):
    """
    True if the URL is a Google News redirect link rather than a direct
    publisher URL. We only IDENTIFY these in this step - decoding them
    into a real CNN/BBC URL is handled in a later step.
    """
    domain = get_domain(url)
    return "news.google.com" in domain


def collect_urls(feed_list, is_valid_domain, target_count, source_name):
    """
    Reads through a list of RSS feed URLs one at a time, collecting unique
    article URLs that belong to the expected source (CNN or BBC), until
    either `target_count` unique URLs have been found or every feed in
    the list has been read.

    `is_valid_domain` is a function like is_cnn_url or is_bbc_url, used to
    check that a link actually belongs to the source we're collecting for.

    Returns a set of unique URLs.
    """
    collected_urls = set()
    rejected_count = 0 
    google_news_link_count = 0

    for feed_url in feed_list:
        if len(collected_urls) >= target_count:
            break

        # Don't let one broken feed stop the whole collection process
        try:
            entries = read_feed(feed_url, verbose=False)
        except Exception as error:
            print(f"Could not read feed {feed_url}: {error}")
            continue

        for entry in entries:
            if len(collected_urls) >= target_count:
                break

            link = entry.get("link", "")
            if not link:
                continue

            if is_google_news_url(link):
                # Identified, but not decoded yet - skip for now
                google_news_link_count += 1
                continue

            if not is_valid_domain(link):
                continue

            if not is_valid_article_url(link):        # <-- ADD THIS
                rejected_count += 1                    # <-- ADD THIS
                continue                               # <-- ADD THIS

            if link in collected_urls:
                continue

            collected_urls.add(link)

            # Print progress every 25 new unique URLs
            if len(collected_urls) % 25 == 0:
                print(f"{source_name} URLs collected: {len(collected_urls)}/{target_count}")

    if google_news_link_count > 0:
        print(f"Note: found {google_news_link_count} Google News link(s) in the {source_name} "
              f"feeds (identified but not decoded in this step).")

    return collected_urls, rejected_count

def write_urls_to_csv(urls, source, filename):
    """
    Writes a set/list of unique URLs to a CSV file with columns:
    id,source,url
    """
    # Make sure the target folder exists (e.g. "data/")
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Sort for a stable, repeatable order between runs, then assign IDs
    unique_urls = sorted(set(urls))

    rows = [
        {"id": i, "source": source, "url": url}
        for i, url in enumerate(unique_urls, start=1)
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "source", "url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{source} URLs saved: {len(rows)}")
    print(f"File: {filename}")

    return rows  # so we can print a preview without re-reading the file


if __name__ == "__main__":
    # --- Collect CNN URLs ---
    print("=== Collecting CNN URLs ===")
    cnn_urls,rejected_count = collect_urls(CNN_RSS_FEEDS, is_cnn_url, TARGET_COUNT, "CNN")

    # --- Collect BBC URLs ---
    print("\n=== Collecting BBC URLs ===")
    bbc_urls,rejected_count = collect_urls(BBC_RSS_FEEDS, is_bbc_url, TARGET_COUNT, "BBC")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Total CNN URLs collected: {len(cnn_urls)}")
    print(f"Total BBC URLs collected: {len(bbc_urls)}")

    # Be honest if the real feeds didn't provide enough entries -
    # we never invent extra URLs to hit the target.
    if len(cnn_urls) < TARGET_COUNT:
        print(f"Note: only {len(cnn_urls)} unique CNN URLs were available from the feeds "
              f"listed in CNN_RSS_FEEDS (target was {TARGET_COUNT}). More feeds or another "
              f"discovery method (e.g. Google News search) would be needed to reach the target.")
    if len(cnn_urls) < 250:
       cnn_urls = collect_cnn_urls_from_google_news(cnn_urls, target_count=250)

    if len(bbc_urls) < TARGET_COUNT:
        print(f"Note: only {len(bbc_urls)} unique BBC URLs were available from the feeds "
              f"listed in BBC_RSS_FEEDS (target was {TARGET_COUNT}). More feeds or another "
              f"discovery method would be needed to reach the target.")

    cnn_url_list = list(cnn_urls)
    bbc_url_list = list(bbc_urls)

    print("\nFirst 5 CNN URLs:")
    for url in cnn_url_list[:5]:
        print(f"- {url}")

    print("\nFirst 5 BBC URLs:")
    for url in bbc_url_list[:5]:
        print(f"- {url}")

    # --- Save both URL sets to CSV ---
cnn_rows = write_urls_to_csv(cnn_urls, "CNN", "data/cnn_urls.csv")
bbc_rows = write_urls_to_csv(bbc_urls, "BBC", "data/bbc_urls.csv")

print("\n=== CSV SAVING COMPLETE ===")
print(f"CNN URLs saved: {len(cnn_rows)}")
print("File: data/cnn_urls.csv")
print(f"BBC URLs saved: {len(bbc_rows)}")
print("File: data/bbc_urls.csv")
print(f"Total URLs saved: {len(cnn_rows) + len(bbc_rows)}")

print("\nFirst 5 CNN rows:")
for row in cnn_rows[:5]:
    print(row)

print("\nFirst 5 BBC rows:")
for row in bbc_rows[:5]:
    print(row)    


