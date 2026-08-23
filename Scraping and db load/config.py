# =============================================================================
#  NEWS PIPELINE CONFIGURATION
# =============================================================================
import os
from zoneinfo import ZoneInfo

# Base Directory & Data Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Sources File (200 sources input file)
SOURCES_FILE = os.path.join(DATA_DIR, "source_counts (1).json")

# Target Limits
ARTICLES_PER_SOURCE = 50       # Fetch 50 articles per source (200 * 50 = 10,000 articles)
MAX_SOURCES = 200              # Total sources to process from sources file

# Date Range Filter for Google News RSS (adjust or leave empty if broad range needed)
AFTER_DATE = "2021-01-01"
BEFORE_DATE = "2026-12-31"

# Concurrency & Performance
RSS_WORKERS = 30               # Parallel threads for RSS fetching
ENRICH_REQUESTS_WORKERS = 80   # Parallel threads for Tier 1 fast HTTP scraping
PHASE2_BROWSERS = 2            # Number of nodriver browser instances
TABS_PER_BROWSER = 3           # Concurrent tabs per nodriver browser
REQUEST_TIMEOUT = 2.5          # Fast timeout in seconds for HTTP requests


# Autosave Intervals
RSS_SAVE_INTERVAL = 10         # Save RSS progress every N sources
ENRICH_SAVE_INTERVAL = 25      # Save enrichment progress every N articles

# File Paths for Intermediate & Final Data
RSS_OUTPUT_FILE = os.path.join(DATA_DIR, "raw_news_articles.json")
RSS_CSV_FILE = os.path.join(DATA_DIR, "raw_news_articles.csv")

ENRICH_OUTPUT_FILE = os.path.join(DATA_DIR, "10000_articles_scraped.json")
ENRICH_CSV_FILE = os.path.join(DATA_DIR, "10000_articles_scraped.csv")

CACHE_FILE = os.path.join(DATA_DIR, "url_cache.json")
FAILED_FILE = os.path.join(DATA_DIR, "failed_articles.json")

# Final Cleaned Output Files
CLEANED_JSON_FILE = os.path.join(DATA_DIR, "10000_articles_clean.json")
CLEANED_CSV_FILE = os.path.join(DATA_DIR, "10000_articles_clean.csv")
SQL_SCHEMA_FILE = os.path.join(BASE_DIR, "init_db.sql")


# Timezones
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# User-Agent and Headers for HTTP requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# Paywalled or Heavy Anti-Bot Sources to Route to Playwright / Special Browsers
PAYWALLED_SOURCES = {
    "bloomberg.com", "bloomberg",
    "wsj.com", "wsj", "wall street journal",
    "ft.com", "financial times",
    "barrons.com", "barron's",
}

# PostgreSQL Database Configuration (Update with your DB credentials when connecting to DB)
POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "news_db",
    "user": "postgres",
    "password": "Usman@1562"
}
