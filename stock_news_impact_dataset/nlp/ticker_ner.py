"""Ticker identification for news articles.

Evidence sources (each yields a confidence; several are combined with a
noisy-OR, so independent agreeing evidence increases confidence):

| method            | evidence                                                        | base conf |
|-------------------|-----------------------------------------------------------------|-----------|
| api_alpaca        | symbol tagged by Alpaca/Benzinga editors                        | 0.85      |
| api_alpha_vantage | symbol in AV ticker_sentiment, scaled by AV relevance_score     | 0.45-0.95 |
| exchange_tag      | "(NASDAQ: AAPL)", "NYSE:IBM"                                    | 0.95      |
| cashtag           | "$AAPL"                                                         | 0.90      |
| paren_ticker      | "Apple (AAPL)"                                                  | 0.80      |
| ner_alias         | company name / alias found in text (spaCy entity-checked)       | 0.60-0.85 |

Every candidate must be in the US-equity reference list built from Alpaca's
asset master (active, NYSE/NASDAQ/AMEX/ARCA/BATS, not an ETF/fund/warrant/unit
unless INCLUDE_ETFS=true). Crypto ("CRYPTO:BTC") and forex ("FOREX:USD") tags
are discarded. Bare upper-case words are NEVER treated as tickers on their own.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from config.settings import Settings
from storage.checkpoint import Checkpoint
from storage.csv_writer import IncrementalCSVWriter, iter_csv_chunks, read_column_set

log = logging.getLogger(__name__)

# Well-known names/brands whose ticker is not recoverable from the legal name,
# or which are too important to leave to the generic alias builder.
CURATED_ALIASES: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN", "amazon.com": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "youtube": "GOOGL", "meta platforms": "META", "meta": "META",
    "facebook": "META", "instagram": "META", "tesla": "TSLA", "berkshire hathaway": "BRK.B", "berkshire": "BRK.B",
    "jpmorgan": "JPM", "jpmorgan chase": "JPM", "jp morgan": "JPM", "visa": "V", "mastercard": "MA",
    "walmart": "WMT", "netflix": "NFLX", "amd": "AMD", "advanced micro devices": "AMD", "intel": "INTC",
    "broadcom": "AVGO", "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE", "qualcomm": "QCOM",
    "micron": "MU", "micron technology": "MU", "ibm": "IBM", "cisco": "CSCO", "palantir": "PLTR",
    "coinbase": "COIN", "boeing": "BA", "disney": "DIS", "walt disney": "DIS", "nike": "NKE",
    "starbucks": "SBUX", "mcdonald's": "MCD", "mcdonalds": "MCD", "coca-cola": "KO", "coca cola": "KO",
    "pepsico": "PEP", "pepsi": "PEP", "pfizer": "PFE", "moderna": "MRNA", "eli lilly": "LLY", "lilly": "LLY",
    "novo nordisk": "NVO", "johnson & johnson": "JNJ", "j&j": "JNJ", "unitedhealth": "UNH",
    "unitedhealth group": "UNH", "exxon": "XOM", "exxonmobil": "XOM", "exxon mobil": "XOM", "chevron": "CVX",
    "goldman sachs": "GS", "goldman": "GS", "morgan stanley": "MS", "bank of america": "BAC",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C", "uber": "UBER", "airbnb": "ABNB", "shopify": "SHOP",
    "snowflake": "SNOW", "super micro computer": "SMCI", "supermicro": "SMCI", "super micro": "SMCI",
    "arm holdings": "ARM", "tsmc": "TSM", "taiwan semiconductor": "TSM", "alibaba": "BABA", "rivian": "RIVN",
    "lucid": "LCID", "ford": "F", "ford motor": "F", "general motors": "GM", "costco": "COST",
    "home depot": "HD", "lowe's": "LOW", "target": "TGT", "crowdstrike": "CRWD", "dell": "DELL",
    "hp inc": "HPQ", "hewlett packard enterprise": "HPE", "spotify": "SPOT", "paypal": "PYPL", "block": "XYZ",
    "robinhood": "HOOD", "microstrategy": "MSTR", "gamestop": "GME", "amc entertainment": "AMC",
    "caterpillar": "CAT", "3m": "MMM", "honeywell": "HON", "lockheed martin": "LMT", "lockheed": "LMT",
    "ups": "UPS", "united parcel service": "UPS", "fedex": "FDX", "at&t": "T", "verizon": "VZ",
    "t-mobile": "TMUS", "comcast": "CMCSA", "merck": "MRK", "abbvie": "ABBV", "bristol-myers squibb": "BMY",
    "bristol myers": "BMY", "amgen": "AMGN", "gilead": "GILD", "regeneron": "REGN", "vertex pharmaceuticals": "VRTX",
    "southwest airlines": "LUV", "delta air lines": "DAL", "delta airlines": "DAL", "united airlines": "UAL",
    "american airlines": "AAL", "carnival": "CCL", "servicenow": "NOW", "intuit": "INTU", "workday": "WDAY",
    "palo alto networks": "PANW", "fortinet": "FTNT", "zscaler": "ZS", "datadog": "DDOG", "mongodb": "MDB",
    "applied materials": "AMAT", "lam research": "LRCX", "kla": "KLAC", "texas instruments": "TXN",
    "marvell": "MRVL", "on semiconductor": "ON", "onsemi": "ON", "asml": "ASML", "sofi": "SOFI",
    "american express": "AXP", "amex": "AXP", "charles schwab": "SCHW", "blackrock": "BLK",
    "blackstone": "BX", "s&p global": "SPGI", "moody's": "MCO", "cvs": "CVS", "cvs health": "CVS",
    "walgreens": "WBA", "humana": "HUM", "cigna": "CI", "abbott": "ABT", "medtronic": "MDT",
    "thermo fisher": "TMO", "danaher": "DHR", "intuitive surgical": "ISRG", "procter & gamble": "PG",
    "p&g": "PG", "colgate": "CL", "philip morris": "PM", "altria": "MO", "mondelez": "MDLZ",
    "kraft heinz": "KHC", "general mills": "GIS", "chipotle": "CMG", "domino's": "DPZ", "yum brands": "YUM",
    "booking holdings": "BKNG", "expedia": "EXPE", "marriott": "MAR", "hilton": "HLT", "doordash": "DASH",
    "lyft": "LYFT", "snap": "SNAP", "snapchat": "SNAP", "pinterest": "PINS", "reddit": "RDDT",
    "roblox": "RBLX", "electronic arts": "EA", "take-two": "TTWO", "warner bros. discovery": "WBD",
    "warner bros discovery": "WBD", "paramount": "PARA", "lockheed martin corp": "LMT", "rtx": "RTX",
    "raytheon": "RTX", "northrop grumman": "NOC", "general dynamics": "GD", "general electric": "GE",
    "ge aerospace": "GE", "ge vernova": "GEV", "ge healthcare": "GEHC", "deere": "DE", "john deere": "DE",
    "conocophillips": "COP", "occidental": "OXY", "occidental petroleum": "OXY", "schlumberger": "SLB",
    "halliburton": "HAL", "nextera": "NEE", "nextera energy": "NEE", "duke energy": "DUK",
    "constellation energy": "CEG", "vistra": "VST", "enphase": "ENPH", "first solar": "FSLR",
    "nike inc": "NKE", "lululemon": "LULU", "ulta": "ULTA", "best buy": "BBY", "dollar general": "DG",
    "dollar tree": "DLTR", "kroger": "KR", "tjx": "TJX", "ross stores": "ROST", "etsy": "ETSY",
    "ebay": "EBAY", "wayfair": "W", "chewy": "CHWY", "peloton": "PTON", "zoom": "ZM", "zoom video": "ZM",
    "docusign": "DOCU", "twilio": "TWLO", "okta": "OKTA", "cloudflare": "NET", "unity software": "U",
    "c3.ai": "AI", "soundhound": "SOUN", "ionq": "IONQ", "rigetti": "RGTI", "marathon digital": "MARA",
    "riot platforms": "RIOT", "nio": "NIO", "xpeng": "XPEV", "li auto": "LI", "baidu": "BIDU", "jd.com": "JD",
    "pdd holdings": "PDD", "temu": "PDD", "sea limited": "SE", "mercadolibre": "MELI", "nu holdings": "NU",
    "toyota": "TM", "honda": "HMC", "sony": "SONY", "shell": "SHEL", "bp": "BP", "astrazeneca": "AZN",
    "novartis": "NVS", "sanofi": "SNY", "gsk": "GSK", "unilever": "UL", "hsbc": "HSBC", "ubs": "UBS",
    "barclays": "BCS", "deutsche bank": "DB", "sap": "SAP", "spotify technology": "SPOT",
    "strategy": "MSTR", "circle": "CRCL", "coreweave": "CRWV", "oklo": "OKLO", "hims & hers": "HIMS",
    "applovin": "APP", "axon": "AXON", "vertiv": "VRT", "arista": "ANET", "arista networks": "ANET",
    "carvana": "CVNA", "affirm": "AFRM", "upstart": "UPST", "draftkings": "DKNG", "celsius": "CELH",
}

# Single words that are companies but also ordinary English / common names:
# only accepted when spaCy tags the span as an ORG.
AMBIGUOUS_ALIASES = {
    "target", "block", "snap", "visa", "shell", "strategy", "circle", "ford", "zoom", "oracle",
    "delta", "carnival", "deere", "hilton", "ups", "sap",
    "arm", "axon", "celsius", "chewy", "affirm", "unity", "lucid", "amex", "gap", "match",
    "square", "chase", "discover", "progressive", "southern", "dominion", "ball", "crown", "carrier",
    "marathon", "lumen", "fox", "news", "general", "american", "united", "first", "global", "national",
    "energy", "capital", "trust", "bank", "health", "technology", "international", "digital", "royal",
}

# Frequent acronyms that are also valid tickers ("artificial intelligence (AI)").
ACRONYM_STOPLIST = {
    "AI", "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "EPS", "GDP", "CPI", "PPI", "FDA", "SEC", "FTC", "DOJ",
    "USA", "US", "UK", "EU", "EV", "EVS", "IT", "ON", "ALL", "ARE", "CAN", "FOR", "NOW", "ONE", "OUT", "BIG",
    "API", "APP", "SAAS", "IOT", "AR", "VR", "PC", "TV", "FED", "FOMC", "OPEC", "NYSE", "NASDAQ", "Q1", "Q2",
    "Q3", "Q4", "YOY", "QOQ", "EBITDA", "GAAP", "ESG", "AGM", "M&A", "LLC", "INC", "LTD", "PLC", "NV", "SA",
    "AG", "SE", "LP", "CO", "CORP", "USD", "EUR", "BTC", "ETH", "IRS", "WHO", "NATO", "OK", "PR", "HR", "RE",
    "GE", "UP", "NEW", "GO", "SO", "AN", "AM", "PM", "BE", "DO", "IS", "OR", "AT", "BY", "TO", "OF", "IN",
}

EXCHANGE_TAG_RE = re.compile(
    r"\b(?:NASDAQ(?:GS|GM|CM)?|Nasdaq(?:GS|GM|CM)?|NYSE(?:\s?American|\s?Arca|\s?MKT)?|AMEX|Cboe(?:\s?BZX)?|BATS)"
    r"\s*:\s*([A-Z]{1,5}(?:\.[A-Z])?)\b")
CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9$])\$([A-Z]{1,5}(?:\.[A-Z])?)\b")
PAREN_TICKER_RE = re.compile(r"\(([A-Z]{2,5}(?:\.[A-Z])?)\)")
TOKEN_RE = re.compile(r"[A-Za-z0-9&][A-Za-z0-9&\-]*(?:[.'][A-Za-z]+)*")

NAME_SUFFIX_RE = re.compile(
    r"\b(Class [A-Z]\b.*|Common Stock|Ordinary Shares?|American Depositary Shares?.*|ADS|Depositary.*|"
    r"Common Shares?|Shares of Beneficial Interest|Subordinate Voting Shares?|New|"
    r"Incorporated|Inc|Corporation|Corp|Company|Co|Holdings?|Group|Ltd|Limited|plc|PLC|N\.?V|S\.?A|AG|SE|"
    r"L\.?P|LLC|Trust|Bancorp(oration)?)\b\.?", re.IGNORECASE)


def clean_company_name(name: str) -> str:
    name = re.sub(r"\(.*?\)", " ", name or "")
    name = re.sub(r"^The\s+", "", name.strip(), flags=re.IGNORECASE)
    prev = None
    while prev != name:
        prev = name
        name = NAME_SUFFIX_RE.sub(" ", name)
        name = re.sub(r"[,\s]+$", "", re.sub(r"\s+", " ", name)).strip(" ,.-")
    return name


def tokenize(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in TOKEN_RE.finditer(text):
        tok, s, e = m.group(0), m.start(), m.end()
        if tok.lower().endswith("'s"):
            tok, e = tok[:-2], e - 2
        out.append((tok, s, e))
    return out


def alias_key(text: str) -> str:
    return " ".join(t.lower() for t, _, _ in tokenize(text))


class TickerReference:
    """US-equity validation list + company-name alias table."""

    def __init__(self, assets: pd.DataFrame, settings: Settings):
        allowed_ex = set(settings.allowed_exchanges)
        df = assets.copy()
        df["is_excluded"] = df["is_excluded"].astype(str).str.lower() == "true"
        excluded_fund = df["exclude_reason"] == "etf_or_fund"
        keep = df["exchange"].isin(allowed_ex) & (~df["is_excluded"] | (settings.include_etfs & excluded_fund))
        self.equities = df[keep]
        self.valid: set[str] = set(self.equities["symbol"])
        self.names: dict[str, str] = dict(zip(self.equities["symbol"], self.equities["name"]))
        self.excluded: dict[str, str] = dict(zip(df.loc[~keep, "symbol"], df.loc[~keep, "exclude_reason"]))

        candidates: dict[str, set[str]] = {}
        for sym, name in self.names.items():
            key = alias_key(clean_company_name(name))
            if key and len(key) >= 3:
                candidates.setdefault(key, set()).add(sym)
        self.aliases: dict[str, tuple[str, str]] = {}  # key -> (symbol, kind)
        for key, syms in candidates.items():
            if len(syms) == 1:
                self.aliases[key] = (next(iter(syms)), "derived")
        for key, sym in CURATED_ALIASES.items():
            if sym in self.valid:
                self.aliases[alias_key(key)] = (sym, "curated")
        self.max_alias_tokens = max((len(k.split()) for k in self.aliases), default=1)
        log.info("Ticker reference: %d valid US equities, %d aliases", len(self.valid), len(self.aliases))

    def company_name(self, symbol: str) -> str:
        return self.names.get(symbol, "")


def normalize_api_symbol(sym: str) -> str | None:
    """Map provider symbols to Alpaca format; drop crypto/forex/foreign tags."""
    sym = (sym or "").strip().upper()
    if not sym or ":" in sym:  # CRYPTO:BTC, FOREX:USD, TSX:XXX ...
        return None
    sym = sym.replace("/", ".").replace("-", ".")
    return sym if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z]{1,2})?", sym) else None


def noisy_or(values: list[float]) -> float:
    p = 1.0
    for v in values:
        p *= (1 - v)
    return round(min(0.99, 1 - p), 4)


@dataclass
class Candidate:
    confs: list[float] = field(default_factory=list)
    methods: set[str] = field(default_factory=set)


class TickerExtractor:
    def __init__(self, reference: TickerReference, settings: Settings):
        self.ref = reference
        self.s = settings
        self.nlp = None
        if settings.use_spacy:
            try:
                import spacy

                self.nlp = spacy.load(settings.spacy_model, disable=["parser", "lemmatizer", "tagger",
                                                                      "attribute_ruler", "senter"])
                log.info("spaCy model %s loaded for entity recognition", settings.spacy_model)
            except Exception as exc:  # noqa: BLE001 - optional dependency
                log.warning("spaCy model %s unavailable (%s). Falling back to alias/pattern matching only. "
                            "Install with: python -m spacy download %s", settings.spacy_model, exc, settings.spacy_model)

    # -------------------------------------------------------------- pieces
    def _api_candidates(self, row: dict, cands: dict[str, Candidate]) -> None:
        for sym in _json_list(row.get("api_tickers_alpaca")):
            s = normalize_api_symbol(sym)
            if s in self.ref.valid:
                cands.setdefault(s, Candidate()).confs.append(0.85)
                cands[s].methods.add("api_alpaca")
        for item in _json_list(row.get("av_ticker_sentiment")):
            if not isinstance(item, dict):
                continue
            s = normalize_api_symbol(item.get("ticker", ""))
            if s in self.ref.valid:
                rel = _float(item.get("relevance_score"), 0.0)
                cands.setdefault(s, Candidate()).confs.append(round(0.45 + 0.5 * min(1.0, rel), 4))
                cands[s].methods.add("api_alpha_vantage")

    def _pattern_candidates(self, text: str, cands: dict[str, Candidate]) -> None:
        for regex, conf, method in ((EXCHANGE_TAG_RE, 0.95, "exchange_tag"), (CASHTAG_RE, 0.90, "cashtag"),
                                    (PAREN_TICKER_RE, 0.80, "paren_ticker")):
            for m in regex.finditer(text):
                sym = m.group(1)
                if method != "exchange_tag" and sym in ACRONYM_STOPLIST:
                    continue
                if sym in self.ref.valid:
                    cands.setdefault(sym, Candidate()).confs.append(conf)
                    cands[sym].methods.add(method)

    def _alias_candidates(self, text: str, in_title: bool, entities: list[tuple[int, int, str]],
                          cands: dict[str, Candidate]) -> None:
        tokens = tokenize(text)
        n = len(tokens)
        i = 0
        while i < n:
            matched = False
            for size in range(min(self.ref.max_alias_tokens, n - i), 0, -1):
                span = tokens[i:i + size]
                if not span[0][0][:1].isupper() and not span[0][0][:1].isdigit():
                    continue  # company names are proper nouns
                key = " ".join(t.lower() for t, _, _ in span)
                hit = self.ref.aliases.get(key)
                if not hit:
                    continue
                sym, kind = hit
                start, end = span[0][1], span[-1][2]
                labels = {lab for s, e, lab in entities if s < end and e > start}
                ambiguous = key in AMBIGUOUS_ALIASES
                multi = size > 1
                distinctive = bool(re.search(r"[a-z][A-Z]|\d|&", span[0][0])) or multi
                if ambiguous:
                    if "ORG" not in labels:
                        continue
                    conf = 0.75
                elif kind == "curated":
                    conf = 0.85 if in_title else 0.75
                elif labels & {"ORG", "PRODUCT", "GPE", "NORP", "FAC"}:
                    conf = 0.75
                elif distinctive and (multi or self.nlp is None):
                    conf = 0.6
                else:
                    continue
                cands.setdefault(sym, Candidate()).confs.append(conf)
                cands[sym].methods.add("ner_alias")
                i += size
                matched = True
                break
            if not matched:
                i += 1

    # ---------------------------------------------------------------- main
    def extract_batch(self, rows: list[dict]) -> list[list[dict]]:
        titles = [str(r.get("title") or "") for r in rows]
        descs = [str(r.get("description") or "") for r in rows]
        if self.nlp is not None:
            title_docs = list(self.nlp.pipe(titles, batch_size=256))
            desc_docs = list(self.nlp.pipe(descs, batch_size=256))
            ents = lambda d: [(e.start_char, e.end_char, e.label_) for e in d.ents]  # noqa: E731
            title_ents = [ents(d) for d in title_docs]
            desc_ents = [ents(d) for d in desc_docs]
        else:
            title_ents = desc_ents = [[] for _ in rows]
        results = []
        for row, title, desc, te, de in zip(rows, titles, descs, title_ents, desc_ents):
            cands: dict[str, Candidate] = {}
            self._api_candidates(row, cands)
            self._pattern_candidates(f"{title}\n{desc}", cands)
            self._alias_candidates(title, True, te, cands)
            self._alias_candidates(desc, False, de, cands)
            found = []
            for sym, c in cands.items():
                conf = noisy_or(c.confs)
                if conf >= self.s.min_ticker_confidence:
                    found.append({"ticker": sym, "ticker_confidence": conf,
                                  "ticker_detection_method": "+".join(sorted(c.methods)),
                                  "company_name": self.ref.company_name(sym)})
            found.sort(key=lambda d: (-d["ticker_confidence"], d["ticker"]))
            results.append(found[: self.s.max_tickers_per_article])
        return results


def _json_list(value) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def av_ticker_info(row: dict, ticker: str) -> dict:
    for item in _json_list(row.get("av_ticker_sentiment")):
        if isinstance(item, dict) and normalize_api_symbol(item.get("ticker", "")) == ticker:
            return {"av_ticker_relevance_score": item.get("relevance_score", ""),
                    "av_ticker_sentiment_score": item.get("ticker_sentiment_score", ""),
                    "av_ticker_sentiment_label": item.get("ticker_sentiment_label", "")}
    return {"av_ticker_relevance_score": "", "av_ticker_sentiment_score": "", "av_ticker_sentiment_label": ""}


# ------------------------------------------------------------------- stage

EVENT_COLUMNS = [
    "event_id", "news_id", "ticker", "company_name", "ticker_confidence", "ticker_detection_method",
    "n_tickers_in_article", "title", "description", "news_published_timestamp_utc", "published_timestamp_raw",
    "news_provider", "news_providers_all", "source", "publisher", "source_domain", "author", "url",
    "alpha_vantage_id", "alpaca_id", "overall_sentiment_score", "overall_sentiment_label",
    "av_ticker_relevance_score", "av_ticker_sentiment_score", "av_ticker_sentiment_label", "topics",
    "published_utc_alpha_vantage", "published_utc_alpaca", "duplicate_count",
]
PROCESSED_COLUMNS = ["news_id", "n_tickers", "processed_at_utc"]


def run_ner(settings: Settings, reference: TickerReference) -> dict:
    cp = Checkpoint(settings.checkpoint_dir / "ner.json")
    sig = settings.section_hash("min_ticker_confidence", "max_tickers_per_article", "include_etfs",
                                "allowed_exchanges", "use_spacy", "spacy_model")
    if cp.get("signature") not in (None, sig):
        log.warning("NER settings changed since the last run. Already-processed articles keep their old tickers; "
                    "run `python main.py --ner --force` to re-extract everything.")
    cp.set("signature", cp.get("signature") or sig)

    events_writer = IncrementalCSVWriter(settings.ticker_events_path, EVENT_COLUMNS, cp.get("events_size"))
    log_path = settings.checkpoint_dir / "ner_processed.csv"
    processed_writer = IncrementalCSVWriter(log_path, PROCESSED_COLUMNS, cp.get("processed_size"))
    done = read_column_set(log_path, "news_id")
    extractor = TickerExtractor(reference, settings)
    log.info("NER: %d articles already processed", len(done))

    from collectors.api_utils import STOP_EVENT
    stats = {"articles_processed_this_run": 0, "events_written_this_run": 0, "status": "completed"}
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    for chunk in iter_csv_chunks(settings.dedup_news_path, settings.chunk_size):
        if STOP_EVENT.is_set():
            stats["status"] = "interrupted"
            break
        chunk = chunk.fillna("")
        chunk = chunk[~chunk["news_id"].isin(done)]
        if not len(chunk):
            continue
        rows = chunk.to_dict("records")
        results = extractor.extract_batch(rows)
        events, processed = [], []
        for row, found in zip(rows, results):
            for f in found:
                ev = {k: row.get(k, "") for k in EVENT_COLUMNS}
                ev.update(f)
                ev.update(av_ticker_info(row, f["ticker"]))
                ev["event_id"] = f"{row['news_id']}_{f['ticker']}"
                ev["n_tickers_in_article"] = len(found)
                events.append(ev)
            processed.append({"news_id": row["news_id"], "n_tickers": len(found), "processed_at_utc": now})
        ev_size = events_writer.write_rows(events) if events else events_writer.size()
        pr_size = processed_writer.write_rows(processed)
        cp.update({"events_size": ev_size, "processed_size": pr_size})
        done.update(r["news_id"] for r in processed)
        stats["articles_processed_this_run"] += len(processed)
        stats["events_written_this_run"] += len(events)
        log.info("NER: +%d articles -> +%d ticker events (total processed %d)", len(processed), len(events), len(done))
    stats["articles_processed_total"] = len(done)
    return stats
