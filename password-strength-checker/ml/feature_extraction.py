"""
Feature extraction for the ML-Based Password Strength Checker.

The SAME functions are used for training (train_model.py) and for live
scoring (app.py), so the feature vector shape always matches the model.

Feature groups (following the project slides):
  1. Basic metrics     - length, upper / lower / digit / symbol counts
  2. Advanced metrics  - Shannon entropy, character-set size, total entropy
  3. Pattern analysis  - repeated characters, sequences, dictionary words,
                         common passwords, leet-speak
"""

import math
import os
from collections import Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# The order of this list defines the order of the feature vector.
FEATURE_NAMES = [
    "length",
    "n_upper",
    "n_lower",
    "n_digits",
    "n_symbols",
    "n_classes",
    "n_unique_chars",
    "unique_ratio",
    "repeated_chars",
    "max_repeat_run",
    "max_sequence_len",
    "entropy_per_char",
    "charset_size",
    "total_entropy_bits",
    "has_dictionary_word",
    "is_common_password",
    "leet_count",
    "trailing_digits",
    "upper_ratio",
    "digit_ratio",
    "symbol_ratio",
]

LEET_MAP = str.maketrans({"@": "a", "4": "a", "0": "o", "1": "i", "!": "i",
                          "3": "e", "$": "s", "5": "s", "7": "t", "8": "b"})
LEET_CHARS = set("@401!3$578")

KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890"]


def _load_lines(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {line.strip().lower() for line in fh if line.strip()}


COMMON_PASSWORDS = _load_lines("common_passwords.txt")
COMMON_WORDS = {w for w in _load_lines("common_words.txt") if len(w) >= 4}


# ---------------------------------------------------------------- helpers
def shannon_entropy(password):
    """Shannon entropy in bits per character."""
    if not password:
        return 0.0
    counts = Counter(password)
    n = len(password)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def charset_size(password):
    """Size of the character pool the password draws from."""
    size = 0
    if any(c.islower() for c in password):
        size += 26
    if any(c.isupper() for c in password):
        size += 26
    if any(c.isdigit() for c in password):
        size += 10
    if any(not c.isalnum() for c in password):
        size += 33  # printable ASCII symbols + space
    return size


def max_repeat_run(password):
    """Longest run of the same character, e.g. 'aaab' -> 3."""
    best = run = 1 if password else 0
    for prev, cur in zip(password, password[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def max_sequence_len(password):
    """
    Longest ascending/descending sequence (abc, 321) or keyboard run (qwe, asd).
    Returns 0 when there is no run of 3+ characters.
    """
    pw = password.lower()
    best = 0

    # ascending / descending by character code
    for step in (1, -1):
        run = 1
        for prev, cur in zip(pw, pw[1:]):
            if ord(cur) - ord(prev) == step and cur.isalnum() and prev.isalnum():
                run += 1
            else:
                run = 1
            if run >= 3:
                best = max(best, run)

    # keyboard rows (forward and backward)
    for row in KEYBOARD_ROWS:
        for line in (row, row[::-1]):
            run = 0
            for i, ch in enumerate(pw):
                # is pw[i-1], pw[i] adjacent in this row?
                if i > 0:
                    a = line.find(pw[i - 1])
                    b = line.find(ch)
                    run = run + 1 if (a != -1 and b == a + 1) else 1
                else:
                    run = 1
                if run >= 3:
                    best = max(best, run)
    return best


def _normalise_leet(password):
    return password.lower().translate(LEET_MAP)


def has_dictionary_word(password):
    """True when a common word is hidden in the password (leet-speak aware)."""
    plain = password.lower()
    decoded = _normalise_leet(password)
    return any(w in plain or w in decoded for w in COMMON_WORDS)


def leet_count(password):
    """Number of leet-style substitution characters that sit between letters."""
    count = 0
    for i, ch in enumerate(password):
        if ch in LEET_CHARS and 0 < i < len(password) - 1:
            if password[i - 1].isalpha() or password[i + 1].isalpha():
                count += 1
    return count


def trailing_digits(password):
    n = 0
    for ch in reversed(password):
        if ch.isdigit():
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------- main API
def extract_features(password):
    """Return an ordered dict of numeric features for one password."""
    if not isinstance(password, str):
        raise TypeError("password must be a string")

    length = len(password)
    n_upper = sum(c.isupper() for c in password)
    n_lower = sum(c.islower() for c in password)
    n_digits = sum(c.isdigit() for c in password)
    n_symbols = length - n_upper - n_lower - n_digits
    n_unique = len(set(password))
    n_classes = sum(x > 0 for x in (n_upper, n_lower, n_digits, n_symbols))
    pool = charset_size(password)
    safe_len = max(length, 1)

    feats = {
        "length": length,
        "n_upper": n_upper,
        "n_lower": n_lower,
        "n_digits": n_digits,
        "n_symbols": n_symbols,
        "n_classes": n_classes,
        "n_unique_chars": n_unique,
        "unique_ratio": n_unique / safe_len,
        "repeated_chars": length - n_unique,
        "max_repeat_run": max_repeat_run(password),
        "max_sequence_len": max_sequence_len(password),
        "entropy_per_char": shannon_entropy(password),
        "charset_size": pool,
        "total_entropy_bits": length * math.log2(pool) if pool else 0.0,
        "has_dictionary_word": int(has_dictionary_word(password)),
        "is_common_password": int(password.lower() in COMMON_PASSWORDS),
        "leet_count": leet_count(password),
        "trailing_digits": trailing_digits(password),
        "upper_ratio": n_upper / safe_len,
        "digit_ratio": n_digits / safe_len,
        "symbol_ratio": n_symbols / safe_len,
    }
    return {name: feats[name] for name in FEATURE_NAMES}


def features_to_vector(feats):
    """Convert the feature dict to a (1, n_features) numpy array."""
    return np.array([[feats[name] for name in FEATURE_NAMES]], dtype=float)


def extract_vector(password):
    return features_to_vector(extract_features(password))


# ------------------------------------------------------------ suggestions
def build_suggestions(password, feats=None):
    """Plain-language tips on how to improve a password."""
    f = feats or extract_features(password)
    tips = []

    if f["is_common_password"]:
        tips.append("This is a very common password. Avoid it completely.")
    if f["length"] < 12:
        tips.append("Use at least 12 characters; longer passwords are much harder to crack.")
    if f["n_upper"] == 0:
        tips.append("Add uppercase letters (A-Z).")
    if f["n_lower"] == 0:
        tips.append("Add lowercase letters (a-z).")
    if f["n_digits"] == 0:
        tips.append("Add numbers (0-9).")
    if f["n_symbols"] == 0:
        tips.append("Add special characters such as ! @ # $ % ^ &.")
    if f["has_dictionary_word"] and not f["is_common_password"]:
        tips.append("Avoid dictionary words or names, even with letters swapped for symbols (P@ssw0rd).")
    if f["max_sequence_len"] >= 3:
        tips.append("Avoid sequences such as abc, 123 or qwerty.")
    if f["max_repeat_run"] >= 3 or (f["unique_ratio"] < 0.6 and f["length"] >= 6):
        tips.append("Avoid repeating the same character many times.")
    if f["trailing_digits"] >= 3 and f["n_classes"] <= 2:
        tips.append("Do not just add numbers at the end of a word; mix them throughout.")
    if not tips:
        tips.append("Great job! Keep this password unique and never reuse it on other sites.")
    return tips
