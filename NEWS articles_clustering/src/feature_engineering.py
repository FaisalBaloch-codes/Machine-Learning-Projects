"""
feature_engineering.py
------------------------
STAGE 1 of this project: builds numeric structural features straight from
the raw Title text. This is the "feature-engineered dataset" that the
clustering stage (Stage 2) will run on.

Features created:
  - title_len            : character length of the title
  - num_digits_count      : count of digit characters
  - question_mark_count   : count of '?' characters
  - punctuation_count     : count of any string.punctuation character
  - sentiment_score        : VADER compound sentiment score (-1 to +1)
  - poi_count             : count of multi-word capitalized phrases
                             (a lightweight proxy for named entities, e.g.
                             "Bank of Queensland", "Philip Morris")
  - has_location          : 1 if a country/major-city keyword from a small
                             gazetteer appears in the title, else 0

These are genuinely engineered, numeric, model-ready features -- exactly
what clustering (Stage 2) needs. URL, Title and Category are identifiers /
labels, not features, and are kept alongside for reference but never fed
into the clustering algorithms.
"""

import re
import string

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

_LOCATION_KEYWORDS = [
    "india", "china", "chinese", "uk", "u.s.", "us", "usa", "america",
    "american", "london", "dubai", "pakistan", "europe", "european",
    "australia", "canada", "japan", "brazil", "italy", "italian",
    "ireland", "israel", "thailand", "germany", "france", "russia",
    "spain", "mexico", "singapore", "korea", "vietnam", "new york",
    "california", "washington", "texas", "wall street",
]

_PROPER_NOUN_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")

ENGINEERED_FEATURE_COLUMNS = [
    "title_len", "num_digits_count", "question_mark_count",
    "punctuation_count", "sentiment_score", "poi_count", "has_location",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the 7 structural feature columns added."""
    df = df.copy()
    titles = df["Title"].astype(str)

    df["title_len"] = titles.str.len()
    df["num_digits_count"] = titles.apply(lambda t: sum(c.isdigit() for c in t))
    df["question_mark_count"] = titles.str.count(r"\?")
    df["punctuation_count"] = titles.apply(lambda t: sum(c in string.punctuation for c in t))
    df["sentiment_score"] = titles.apply(lambda t: _sia.polarity_scores(t)["compound"])
    df["poi_count"] = titles.apply(lambda t: len(_PROPER_NOUN_PHRASE.findall(t)))
    df["has_location"] = titles.str.lower().apply(
        lambda t: int(any(loc in t for loc in _LOCATION_KEYWORDS))
    )
    return df
