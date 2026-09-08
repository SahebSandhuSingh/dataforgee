"""
Deterministic expected phoneme lookup and normalization.

Provides get_expected_phonemes(word) -> list[str]
which returns a normalized list of ARPAbet phoneme tokens (e.g. ['TH', 'R', 'IY']).

Dictionary source:
  - Primary: CMU Pronouncing Dictionary (deterministic reference table for target vocab)
  - Fallback: Deterministic phonetic transcription rules for unknown words

Normalization rules:
  - Input words are stripped of punctuation and converted to lowercase.
  - ARPAbet stress numbers (0, 1, 2) are stripped so all phonemes are clean tokens (e.g., 'IY1' -> 'IY').

Known limitations:
  - Homographs (e.g., 'read' present vs past) only map to the most common pronunciation.
  - Regional variations (e.g., General American vs British RP) default to General American.
"""

import re
from typing import List, Optional

# Deterministic reference vocabulary based on CMUdict (stress numbers stripped)
CMU_REFERENCE_DICT = {
    "three": ["TH", "R", "IY"],
    "think": ["TH", "IH", "NG", "K"],
    "this": ["DH", "IH", "S"],
    "that": ["DH", "AE", "T"],
    "ship": ["SH", "IH", "P"],
    "sheep": ["SH", "IY", "P"],
    "rice": ["R", "AY", "S"],
    "light": ["L", "AY", "T"],
    "right": ["R", "AY", "T"],
    "sound": ["S", "AW", "N", "D"],
    "voice": ["V", "OY", "S"],
    "word": ["W", "ER", "D"],
    "slow": ["S", "L", "OW"],
    "stop": ["S", "T", "AA", "P"],
}

# Rule-based fallback phoneme mappings for grapheme clusters
_RULES = [
    (r"th", ["TH"]),
    (r"sh", ["SH"]),
    (r"ch", ["CH"]),
    (r"ng", ["NG"]),
    (r"ee|ea", ["IY"]),
    (r"oo", ["UW"]),
    (r"igh|ight", ["AY", "T"]),
    (r"ice|ise", ["AY", "S"]),
    (r"a", ["AE"]),
    (r"e", ["EH"]),
    (r"i", ["IH"]),
    (r"o", ["AA"]),
    (r"u", ["AH"]),
    (r"b", ["B"]),
    (r"c|k", ["K"]),
    (r"d", ["D"]),
    (r"f", ["F"]),
    (r"g", ["G"]),
    (r"h", ["HH"]),
    (r"j", ["JH"]),
    (r"l", ["L"]),
    (r"m", ["M"]),
    (r"n", ["N"]),
    (r"p", ["P"]),
    (r"r", ["R"]),
    (r"s", ["S"]),
    (r"t", ["T"]),
    (r"v", ["V"]),
    (r"w", ["W"]),
    (r"y", ["Y"]),
    (r"z", ["Z"]),
]


def normalize_word(word: str) -> str:
    """Strip punctuation and whitespace, lowercase."""
    return re.sub(r"[^a-zA-Z]", "", word).lower()


def normalize_phoneme(phoneme: str) -> str:
    """Normalize a phoneme symbol by stripping stress digits and capitalizing."""
    return re.sub(r"\d+", "", phoneme).strip().upper()


def has_pronunciation(word: str) -> bool:
    """
    Check if a word has reliable pronunciation data (in CMUdict or reference dict).
    Returns False for words that would only get rule-based approximations.
    """
    clean_word = normalize_word(word)
    if not clean_word:
        return False

    if clean_word in CMU_REFERENCE_DICT:
        return True

    try:
        import cmudict
        d = cmudict.dict()
        if clean_word in d:
            return True
    except (ImportError, Exception):
        pass

    return False


def safe_lookup(word: str) -> Optional[List[str]]:
    """
    Safe phoneme lookup that returns None when no reliable pronunciation
    data exists. Used by the drill manager to detect dictionary failures
    and provide user-friendly fallback messages.

    Returns:
        List of phonemes if found, None if no reliable data.
    """
    clean_word = normalize_word(word)
    if not clean_word:
        return None

    if clean_word in CMU_REFERENCE_DICT:
        return [normalize_phoneme(p) for p in CMU_REFERENCE_DICT[clean_word]]

    try:
        import cmudict
        d = cmudict.dict()
        if clean_word in d:
            cmu_phonemes = d[clean_word][0]
            return [normalize_phoneme(p) for p in cmu_phonemes]
    except (ImportError, Exception):
        pass

    return None


def get_expected_phonemes(word: str) -> List[str]:
    """
    Look up the expected phoneme sequence for a given word.

    Returns a list of normalized ARPAbet phoneme strings.
    Never raises an exception; falls back to deterministic rule-based mapping if not in CMUdict.
    """
    clean_word = normalize_word(word)
    if not clean_word:
        return []

    # 1. Fast path: reference dictionary
    if clean_word in CMU_REFERENCE_DICT:
        return [normalize_phoneme(p) for p in CMU_REFERENCE_DICT[clean_word]]

    # 2. CMUDict lookup if nltk/cmudict package is installed
    try:
        import cmudict
        d = cmudict.dict()
        if clean_word in d:
            cmu_phonemes = d[clean_word][0]
            return [normalize_phoneme(p) for p in cmu_phonemes]
    except (ImportError, Exception):
        pass

    # 3. Deterministic rule-based fallback
    phonemes = []
    remaining = clean_word
    while remaining:
        matched = False
        for pattern, mapped in _RULES:
            match = re.match(r"^(" + pattern + ")", remaining)
            if match:
                phonemes.extend(mapped)
                remaining = remaining[len(match.group(0)):]
                matched = True
                break
        if not matched:
            remaining = remaining[1:]  # skip unhandled char

    return [normalize_phoneme(p) for p in phonemes]
