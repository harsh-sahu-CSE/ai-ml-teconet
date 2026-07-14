from __future__ import annotations

import re
import string
import unicodedata

import emoji
import contractions
import nltk
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resources() -> None:
    resources: list[str] = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for pkg in resources:
        nltk.download(pkg, quiet=True)


_ensure_nltk_resources()

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS: frozenset[str] = frozenset(stopwords.words("english"))

_SLANG_MAP: dict[str, str] = {
    "omg": "oh my god",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "brb": "be right back",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "smh": "shaking my head",
    "idk": "i do not know",
    "irl": "in real life",
    "fyi": "for your information",
    "btw": "by the way",
    "gr8": "great",
    "luv": "love",
    "ur": "your",
    "u": "you",
    "r": "are",
    "thx": "thanks",
    "ty": "thank you",
    "np": "no problem",
    "nvm": "never mind",
    "ikr": "i know right",
    "af": "as fuck",
    "rn": "right now",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "cya": "see you",
    "ttyl": "talk to you later",
    "bff": "best friend forever",
    "gg": "good game",
    "wp": "well played",
    "nt": "nice try",
    "ez": "easy",
    "w/": "with",
    "w/o": "without",
    "b4": "before",
    "2day": "today",
    "2moro": "tomorrow",
    "4ever": "forever",
    "xoxo": "hugs and kisses",
}

# compile regex patterns once at import time for better performance
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+|\S+\.\w{2,4}/\S*")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_HASHTAG = re.compile(r"#(\w+)")
_RE_MENTION = re.compile(r"@\w+")
_RE_REPEATED_CHARS = re.compile(r"(.)\1{2,}")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_NON_ASCII_PUNCT = re.compile(r"[^\w\s\'-]")


def remove_html_tags(text: str) -> str:
    return _RE_HTML_TAG.sub(" ", text)


def remove_urls(text: str) -> str:
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    return text


def normalize_hashtags_and_mentions(text: str) -> str:
    # keep the word behind # but drop the symbol, remove @mentions entirely
    text = _RE_HASHTAG.sub(r"\1", text)
    text = _RE_MENTION.sub(" ", text)
    return text


def normalize_emojis(text: str) -> str:
    # convert emoji to readable text e.g. 🔥 -> fire
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ")
    return text


def expand_contractions(text: str) -> str:
    return contractions.fix(text)


def expand_slang(text: str) -> str:
    tokens: list[str] = text.split()
    expanded: list[str] = [_SLANG_MAP.get(token.lower(), token) for token in tokens]
    return " ".join(expanded)


def reduce_repeated_characters(text: str) -> str:
    # sooooo -> soo, keeps double letters like "good" intact
    return _RE_REPEATED_CHARS.sub(r"\1\1", text)


def remove_punctuation(text: str) -> str:
    return _RE_NON_ASCII_PUNCT.sub(" ", text)


def normalize_unicode(text: str) -> str:
    normalized: str = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", errors="ignore").decode("ascii")


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_extra_whitespace(text: str) -> str:
    return _RE_WHITESPACE.sub(" ", text).strip()


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [tok for tok in tokens if tok not in _STOP_WORDS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [_LEMMATIZER.lemmatize(tok) for tok in tokens]


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang in ("en", "unknown"):
        return text
    try:
        translated: str = GoogleTranslator(source=source_lang, target="en").translate(text)
        return translated if translated else text
    except Exception as exc:
        raise RuntimeError(f"Translation failed for language '{source_lang}': {exc}") from exc


class PreprocessingResult:
    def __init__(
        self,
        original_text: str,
        detected_lang: str,
        translated_text: str,
        cleaned_text: str,
        tokens: list[str],
    ) -> None:
        self.original_text: str = original_text
        self.detected_lang: str = detected_lang
        self.translated_text: str = translated_text
        self.cleaned_text: str = cleaned_text
        self.tokens: list[str] = tokens

    def __repr__(self) -> str:
        return (
            f"PreprocessingResult("
            f"lang={self.detected_lang!r}, "
            f"tokens={self.tokens[:8]}{'...' if len(self.tokens) > 8 else ''})"
        )


def preprocess(
    text: str,
    *,
    remove_stops: bool = True,
    lemmatize: bool = True,
    translate: bool = True,
) -> PreprocessingResult:
    if not isinstance(text, str):
        raise ValueError(f"Expected str, got {type(text).__name__!r}")
    if not text.strip():
        raise ValueError("Input text must not be empty.")

    original: str = text

    detected_lang: str = detect_language(text)

    if translate:
        text = translate_to_english(text, detected_lang)
    translated: str = text

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = normalize_hashtags_and_mentions(text)
    text = normalize_emojis(text)
    text = expand_contractions(text)
    text = expand_slang(text)
    text = reduce_repeated_characters(text)
    text = normalize_unicode(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)

    cleaned_text: str = text

    tokens: list[str] = word_tokenize(text)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    return PreprocessingResult(
        original_text=original,
        detected_lang=detected_lang,
        translated_text=translated,
        cleaned_text=cleaned_text,
        tokens=tokens,
    )


def preprocess_batch(
    texts: list[str],
    *,
    remove_stops: bool = True,
    lemmatize: bool = True,
    translate: bool = True,
) -> list[PreprocessingResult]:
    if not texts:
        raise ValueError("texts list must not be empty.")
    return [
        preprocess(t, remove_stops=remove_stops, lemmatize=lemmatize, translate=translate)
        for t in texts
    ]


if __name__ == "__main__":
    samples: list[str] = [
        "OMG this product is sooooo amazing!! 🔥🔥 luv it tbh 💯",
        "<b>Check out</b> https://example.com for gr8 deals!! @user #sale",
        "Ce produit est absolument magnifique, je l'adore!",
        "Worst. Purchase. EVER 😡😡😡 Total waste of money broooo",
    ]

    for i, sample in enumerate(samples, 1):
        result: PreprocessingResult = preprocess(sample)
        print(f"\n[{i}] Original : {result.original_text}")
        print(f"    Language : {result.detected_lang}")
        print(f"    Cleaned  : {result.cleaned_text}")
        print(f"    Tokens   : {result.tokens}")
