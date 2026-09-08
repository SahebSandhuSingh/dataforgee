"""
Rime TTS isolated phoneme synthesis and speed tier formatting.

Provides:
  - build_isolated_phoneme_pronunciation(phoneme) -> str
  - get_speed_alpha(speed_tier) -> float
  - format_drill_playback(phoneme, word, speed_tier) -> dict

Converts ARPAbet phonemes into Rime mistv3 custom-pronunciation representation
so Rime demonstrates the isolated sound itself rather than speaking a full word.
"""

from typing import Any, Dict, Optional
from config import RIME_MODEL, RIME_SPEAKER, SPEED_TIERS
from phoneme_dict import normalize_phoneme

# Rime Mist v3 phonetic representations for isolated phonemes
# Maps ARPAbet phonemes to Rime custom pronunciation / phonetic sound prompts
RIME_PHONEME_MAP = {
    # Fricatives
    "TH": "{TH}",       # voiceless dental fricative /θ/ as in 'three', 'think'
    "DH": "{DH}",       # voiced dental fricative /ð/ as in 'this', 'that'
    "SH": "{SH}",       # voiceless postalveolar fricative /ʃ/ as in 'ship', 'sheep'
    "ZH": "{ZH}",       # voiced postalveolar fricative /ʒ/ as in 'measure'
    "S":  "{S}",        # voiceless alveolar fricative /s/ as in 'rice'
    "Z":  "{Z}",        # voiced alveolar fricative /z/ as in 'zoo'
    "F":  "{F}",        # voiceless labiodental fricative /f/
    "V":  "{V}",        # voiced labiodental fricative /v/
    "HH": "{HH}",       # voiceless glottal fricative /h/

    # Stops / Plosives
    "P":  "{P}",        # voiceless bilabial stop /p/
    "B":  "{B}",        # voiced bilabial stop /b/
    "T":  "{T}",        # voiceless alveolar stop /t/
    "D":  "{D}",        # voiced alveolar stop /d/
    "K":  "{K}",        # voiceless velar stop /k/
    "G":  "{G}",        # voiced velar stop /g/

    # Liquids & Glides
    "R":  "{R}",        # alveolar approximant /r/ as in 'right', 'rice'
    "L":  "{L}",        # alveolar lateral approximant /l/ as in 'light'
    "W":  "{W}",        # labio-velar approximant /w/
    "Y":  "{Y}",        # palatal approximant /j/

    # Nasals
    "M":  "{M}",        # bilabial nasal /m/
    "N":  "{N}",        # alveolar nasal /n/
    "NG": "{NG}",       # velar nasal /ŋ/ as in 'sing', 'think'

    # Affricates
    "CH": "{CH}",       # voiceless postalveolar affricate /tʃ/
    "JH": "{JH}",       # voiced postalveolar affricate /dʒ/

    # Vowels
    "IY": "{IY}",       # close front unrounded vowel /iː/ as in 'sheep', 'three'
    "IH": "{IH}",       # near-close near-front unrounded vowel /ɪ/ as in 'ship'
    "EY": "{EY}",       # /eɪ/
    "EH": "{EH}",       # /ɛ/
    "AE": "{AE}",       # /æ/
    "AA": "{AA}",       # /ɑː/
    "AO": "{AO}",       # /ɔː/
    "OW": "{OW}",       # /oʊ/ as in 'slow'
    "UH": "{UH}",       # /ʊ/
    "UW": "{UW}",       # /uː/
    "AH": "{AH}",       # /ʌ/
    "ER": "{ER}",       # /ɜːr/
    "AY": "{AY}",       # /aɪ/ as in 'right', 'rice', 'light'
    "AW": "{AW}",       # /aʊ/
    "OY": "{OY}",       # /ɔɪ/
}


def build_isolated_phoneme_pronunciation(phoneme: str) -> str:
    """
    Convert a standardized ARPAbet phoneme into the Rime mistv3
    custom-pronunciation format for isolated sound demonstration.
    """
    clean_phoneme = normalize_phoneme(phoneme)
    if not clean_phoneme:
        return ""

    if clean_phoneme in RIME_PHONEME_MAP:
        return RIME_PHONEME_MAP[clean_phoneme]

    # Fallback to standard Rime bracketed notation
    return f"{{{clean_phoneme}}}"


def get_speed_alpha(speed_tier: str) -> float:
    """Return the Rime speed_alpha value for a given speed tier."""
    return SPEED_TIERS.get(speed_tier.lower(), 1.0)


def format_drill_playback(
    phoneme: Optional[str],
    word: str,
    speed_tier: str = "slow",
) -> Dict[str, Any]:
    """
    Format full drill sequence for Rime TTS:
      1. Isolated phoneme sound
      2. Brief pause
      3. Slowed full word
    """
    speed_alpha = get_speed_alpha(speed_tier)
    isolated_repr = build_isolated_phoneme_pronunciation(phoneme) if phoneme else ""

    # Constructed speech script for the two-stage demonstration
    if isolated_repr and word:
        # Prompt sequence: Sound demonstration followed by pause and target word
        spoken_script = f"{isolated_repr} ... {word}"
    elif word:
        spoken_script = word
    else:
        spoken_script = isolated_repr

    return {
        "phoneme": phoneme,
        "word": word,
        "isolated_repr": isolated_repr,
        "spoken_script": spoken_script,
        "speed_tier": speed_tier,
        "speed_alpha": speed_alpha,
        "model": RIME_MODEL,
        "speaker": RIME_SPEAKER,
    }
