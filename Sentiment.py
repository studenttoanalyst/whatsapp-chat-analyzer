"""
sentiment.py
============
AI Sentiment Analysis module for WhatsApp Chat Analyzer.

Strategy: Rule-based pipeline using VADER (Valence Aware Dictionary and
sEntiment Reasoner) as the backbone, extended with a custom Roman Urdu /
Hinglish lexicon so that mixed-language chats (English + Roman Urdu) are
handled gracefully without requiring any ML training data or GPU.

Why VADER?
- Zero training required — works out of the box
- Handles emojis, punctuation emphasis (!!!), capitalisation (GREAT)
- Fast enough to score thousands of messages in under a second
- pip-installable, no large model downloads
- Pairs perfectly with Streamlit's single-file deployment model

Architecture
------------
preprocess_message()   → cleans a single raw message string
get_sentiment_label()  → returns Positive / Negative / Neutral
analyze_sentiment()    → main entry: scores every row in the dataframe
get_sentiment_stats()  → aggregates counts, per-user scores, word lists
get_sentiment_summary()→ generates a plain-English chat mood summary
"""

import re
import string
from collections import Counter

import emoji
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Custom Roman Urdu / Hinglish sentiment lexicon
# Positive words mapped to +1, negative to -1.
# These are the most frequent tokens in Pakistani / Indian WhatsApp chats.
# ---------------------------------------------------------------------------
ROMAN_URDU_LEXICON: dict[str, float] = {
    # ── Positive ──────────────────────────────────────────────────────────
    "mashallah": 2.5, "alhamdulillah": 2.5, "subhanallah": 2.5,
    "mubarak": 2.0, "mubarakbad": 2.0, "khushi": 1.8, "pyar": 1.8,
    "mohabbat": 1.8, "achi": 1.5, "acha": 1.5, "accha": 1.5,
    "theek": 1.0, "shukriya": 1.8, "shukria": 1.8, "jazakallah": 2.5,
    "zindagi": 1.2, "khubsurat": 2.0, "sundar": 1.8, "behtareen": 2.2,
    "lajawaab": 2.3, "kamaal": 2.0, "wah": 1.5, "waah": 1.5,
    "mast": 1.5, "zabardast": 2.2, "jannat": 2.0, "khush": 1.8,
    "dilchasp": 1.6, "umda": 1.8, "pyara": 1.8, "pyari": 1.8,
    "meherbani": 1.8, "dua": 1.5, "barkat": 1.8, "success": 1.5,
    "khoobsurat": 2.0, "shaandaar": 2.2, "bahut acha": 2.0,
    "bilkul": 0.8, "zaroor": 0.5, "haha": 1.0, "hahaha": 1.2,
    "lol": 1.0, "lmao": 1.2, "rofl": 1.2, "xd": 1.0,
    # ── Negative ──────────────────────────────────────────────────────────
    "bura": -1.5, "buri": -1.5, "ganda": -1.8, "gandi": -1.8,
    "ghalt": -1.5, "galat": -1.5, "takleef": -1.8, "dard": -1.8,
    "afsos": -2.0, "gham": -2.0, "rona": -1.5, "mushkil": -1.2,
    "pareshaan": -1.8, "tension": -1.5, "gussa": -1.8,
    "naraaz": -1.8, "bura laga": -2.0, "dukh": -2.0,
    "barbad": -2.5, "kharab": -1.8, "bekar": -1.8,
    "faltu": -1.5, "wahiyat": -2.0, "bakwas": -2.0,
    "jhoot": -2.0, "jhoothi": -2.0, "dhoka": -2.5,
    "zulm": -2.5, "bimaar": -1.5, "mareez": -1.2,
    "problem": -1.0, "issue": -0.8, "fail": -1.8, "failed": -1.8,
}

# ---------------------------------------------------------------------------
# Compile patterns once at module load (performance)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[^\w\s]")     # keeps alphanumeric + whitespace
_WHITESPACE_RE = re.compile(r"\s+")

# Messages that carry no semantic content
_SKIP_PATTERNS = {
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "missed voice call",
    "missed video call",
}

# ---------------------------------------------------------------------------
# Module-level VADER analyser instance (thread-safe for Streamlit)
# ---------------------------------------------------------------------------
_analyser = SentimentIntensityAnalyzer()

# Inject our Roman Urdu lexicon into VADER's internal lexicon dict
_analyser.lexicon.update(ROMAN_URDU_LEXICON)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def preprocess_message(text: str) -> str:
    """
    Clean a raw WhatsApp message for sentiment scoring.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Convert emojis to their CLDR text descriptions
       (e.g. 😊 → "smiling face with smiling eyes")
    4. Strip punctuation
    5. Collapse whitespace

    Parameters
    ----------
    text : str
        Raw message string.

    Returns
    -------
    str
        Cleaned text ready for sentiment analysis.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub("", text)

    # Replace emojis with their textual description — VADER can then score
    # words like "smiling" positively.
    text = emoji.demojize(text, delimiters=(" ", " "))

    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def _should_skip(raw: str) -> bool:
    """Return True for media placeholders and system-like messages."""
    lower = raw.lower().strip()
    for pattern in _SKIP_PATTERNS:
        if pattern in lower:
            return True
    return False


def get_sentiment_label(compound_score: float) -> str:
    """
    Convert VADER compound score → human-readable label.

    VADER standard thresholds:
      compound >=  0.05  →  Positive
      compound <= -0.05  →  Negative
      otherwise          →  Neutral

    Parameters
    ----------
    compound_score : float
        VADER compound score in [-1, 1].

    Returns
    -------
    str
        'Positive', 'Negative', or 'Neutral'
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every text message in *df* and return an enriched copy.

    Skips media-only messages (compound = 0, label = 'Neutral').

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed WhatsApp dataframe (from Preprocessing.preprocess).
        Must have a 'message' column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with three new columns:
          - 'clean_message'   : preprocessed text
          - 'compound'        : VADER compound score [-1, 1]
          - 'sentiment'       : 'Positive' / 'Negative' / 'Neutral'
    """
    if df.empty or "message" not in df.columns:
        return df.copy()

    result = df.copy()

    clean_messages: list[str] = []
    compound_scores: list[float] = []
    labels: list[str] = []

    for raw in result["message"]:
        if _should_skip(raw):
            clean_messages.append("")
            compound_scores.append(0.0)
            labels.append("Neutral")
            continue

        cleaned = preprocess_message(raw)

        if not cleaned:
            clean_messages.append("")
            compound_scores.append(0.0)
            labels.append("Neutral")
            continue

        scores = _analyser.polarity_scores(cleaned)
        c = scores["compound"]

        clean_messages.append(cleaned)
        compound_scores.append(c)
        labels.append(get_sentiment_label(c))

    result["clean_message"] = clean_messages
    result["compound"] = compound_scores
    result["sentiment"] = labels

    return result


def get_sentiment_stats(selected_user: str, df: pd.DataFrame) -> dict:
    """
    Compute aggregated sentiment statistics for Streamlit widgets.

    Parameters
    ----------
    selected_user : str
        'Overall' or a specific user name.
    df : pd.DataFrame
        Sentiment-enriched dataframe (output of analyze_sentiment).

    Returns
    -------
    dict with keys
    ─────────────
    counts          : dict {'Positive': int, 'Negative': int, 'Neutral': int}
    sentiment_over_time : pd.DataFrame [only_date, sentiment, count]
    user_scores     : pd.DataFrame [user, mean_compound, message_count]
                       (None when selected_user != 'Overall')
    top_positive_words : list[(word, freq)]  top-20
    top_negative_words : list[(word, freq)]  top-20
    """
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    # Guard: empty after filter
    if df.empty:
        return {
            "counts": {"Positive": 0, "Negative": 0, "Neutral": 0},
            "sentiment_over_time": pd.DataFrame(),
            "user_scores": None,
            "top_positive_words": [],
            "top_negative_words": [],
        }

    # ── Sentiment counts ─────────────────────────────────────────────────
    counts = (
        df["sentiment"]
        .value_counts()
        .reindex(["Positive", "Negative", "Neutral"], fill_value=0)
        .to_dict()
    )

    # ── Sentiment over time ───────────────────────────────────────────────
    sot = (
        df.groupby(["only_date", "sentiment"])["message"]
        .count()
        .reset_index()
        .rename(columns={"message": "count"})
    )

    # ── Per-user scores (Overall only) ────────────────────────────────────
    user_scores = None
    if selected_user == "Overall":
        user_scores = (
            df.groupby("user")
            .agg(
                mean_compound=("compound", "mean"),
                message_count=("compound", "count"),
            )
            .reset_index()
            .sort_values("mean_compound", ascending=False)
        )
        user_scores["mean_compound"] = user_scores["mean_compound"].round(3)

    # ── Word-level sentiment ──────────────────────────────────────────────
    pos_words = _top_sentiment_words(df[df["sentiment"] == "Positive"], 20)
    neg_words = _top_sentiment_words(df[df["sentiment"] == "Negative"], 20)

    return {
        "counts": counts,
        "sentiment_over_time": sot,
        "user_scores": user_scores,
        "top_positive_words": pos_words,
        "top_negative_words": neg_words,
    }


def _top_sentiment_words(sub_df: pd.DataFrame, n: int = 20) -> list[tuple[str, int]]:
    """Extract the most common words from a sentiment-filtered sub-dataframe."""
    # Load stop words if available
    try:
        with open("stop_hinglish.txt", "r", encoding="utf-8") as f:
            stop_words = set(f.read().split())
    except FileNotFoundError:
        stop_words = set()

    # Add common English stop words that may not be in the Hinglish list
    stop_words.update(
        {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "not", "with", "this", "that", "i",
            "me", "my", "you", "your", "we", "our", "they", "their",
            "he", "she", "his", "her", "was", "are", "be", "been",
            "have", "has", "had", "do", "did", "will", "would", "could",
            "should", "can", "may", "might", "shall", "so", "if", "as",
        }
    )

    words: list[str] = []
    for text in sub_df["clean_message"].dropna():
        for word in text.split():
            if len(word) > 2 and word not in stop_words:
                words.append(word)

    return Counter(words).most_common(n)


def get_sentiment_summary(counts: dict, user_scores: pd.DataFrame | None) -> str:
    """
    Generate a plain-English one-paragraph chat mood summary.

    Parameters
    ----------
    counts : dict
        {'Positive': int, 'Negative': int, 'Neutral': int}
    user_scores : pd.DataFrame or None
        Per-user sentiment scores (Overall mode only).

    Returns
    -------
    str
        A descriptive summary sentence.
    """
    total = sum(counts.values())
    if total == 0:
        return "No analysable messages found."

    pos_pct = round(counts["Positive"] / total * 100, 1)
    neg_pct = round(counts["Negative"] / total * 100, 1)
    neu_pct = round(counts["Neutral"] / total * 100, 1)

    # Determine dominant mood
    dominant = max(counts, key=counts.get)

    if dominant == "Positive":
        mood_line = f"Overall, this chat is **mostly positive** 😊 with {pos_pct}% upbeat messages."
    elif dominant == "Negative":
        mood_line = f"Overall, this chat leans **negative** 😟 — {neg_pct}% of messages carry a negative tone."
    else:
        mood_line = f"Overall, this chat is **largely neutral** 😐 ({neu_pct}% neutral messages)."

    # Add secondary observation
    if neg_pct > 20:
        secondary = " There are notable negative spikes worth attention."
    elif pos_pct > 60:
        secondary = " The group maintains a very warm and supportive tone."
    else:
        secondary = f" It has a balanced mix: {pos_pct}% positive, {neg_pct}% negative, {neu_pct}% neutral."

    # Add top-user insight
    user_line = ""
    if user_scores is not None and not user_scores.empty and len(user_scores) >= 2:
        most_positive = user_scores.iloc[0]["user"]
        most_negative = user_scores.iloc[-1]["user"]
        user_line = f" **{most_positive}** is the most positive contributor, while **{most_negative}** tends to be more critical."

    return mood_line + secondary + user_line