"""
Generate a labelled password dataset (data/password_dataset.csv).

WHY THIS FILE EXISTS
--------------------
The project slides use the Kaggle "Password Strength Classifier" dataset
(000webhost leak). That file cannot be bundled or downloaded automatically,
so this script builds a *synthetic* labelled dataset instead, so the project
runs end-to-end out of the box.

  * The passwords are generated, not leaked from real accounts.
  * The labels (weak / medium / strong) come from a documented scoring policy
    (see `policy_score` below), NOT from real cracking experiments.

If you have the real Kaggle file, skip this script and run:
    python ml/train_model.py --data path/to/data.csv
(columns: password, strength where strength is 0/1/2 or weak/medium/strong)

Usage:
    python ml/generate_dataset.py                # default size
    python ml/generate_dataset.py --size 80000   # custom size
"""

import argparse
import math
import os
import random
import string
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.feature_extraction import (  # noqa: E402
    DATA_DIR,
    COMMON_PASSWORDS,
    COMMON_WORDS,
    extract_features,
)

SYMBOLS = "!@#$%^&*()-_=+[]{};:,.<>?/~"
LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}
KEYBOARD_PATTERNS = ["qwerty", "asdfgh", "zxcvbn", "qazwsx", "1q2w3e", "qwertyuiop",
                     "asdfghjkl", "zxcvbnm", "123456", "654321", "abcdef", "abcd1234"]
EXTRA_WORDS = ["blue", "river", "stone", "cloud", "tiger", "maple", "ocean", "piano",
               "rocket", "garden", "violet", "copper", "falcon", "harbor", "meadow",
               "walnut", "yellow", "window", "bridge", "castle", "planet", "silver",
               "thunder", "marble", "lantern", "puzzle", "velvet", "orchid", "canyon"]


# ------------------------------------------------------------------ labels
def policy_score(pw):
    """
    Documented labelling policy (max score = 9).

      length         <6:0   6-7:1   8-11:2   12-15:3   16+:4
      variety        +1 for each extra character class used (max +3)
      entropy        +1 if total entropy >= 60 bits, +1 more if >= 80 bits
      penalties      common password  -> score forced to 0
                     dictionary word  -2
                     sequence (abc/123/qwe)  -1
                     run of 3+ identical characters  -1
                     fewer than 55% unique characters (and < 10 distinct chars)  -1

      weak   : score <= 3
      medium : score 4-6
      strong : score >= 7
    """
    f = extract_features(pw)
    if f["is_common_password"]:
        return 0

    n = f["length"]
    score = 0 if n < 6 else 1 if n < 8 else 2 if n < 12 else 3 if n < 16 else 4
    score += max(f["n_classes"] - 1, 0)
    if f["total_entropy_bits"] >= 60:
        score += 1
    if f["total_entropy_bits"] >= 80:
        score += 1
    if f["has_dictionary_word"]:
        score -= 2
    if f["max_sequence_len"] >= 3:
        score -= 1
    if f["max_repeat_run"] >= 3:
        score -= 1
    if f["unique_ratio"] < 0.55 and f["n_unique_chars"] < 10:
        score -= 1
    return max(score, 0)


def policy_label(pw):
    s = policy_score(pw)
    return "weak" if s <= 3 else "medium" if s <= 6 else "strong"


# --------------------------------------------------------------- generators
def _rand_chars(n, pool):
    return "".join(random.choice(pool) for _ in range(n))


def _words():
    return list(COMMON_WORDS) + EXTRA_WORDS


def _leetify(word, prob=0.6):
    return "".join(LEET.get(c, c) if random.random() < prob else c for c in word)


def _decorate_case(word):
    r = random.random()
    if r < 0.4:
        return word.capitalize()
    if r < 0.5:
        return word.upper()
    return word


def gen_common_variant():
    base = random.choice(sorted(COMMON_PASSWORDS))
    r = random.random()
    if r < 0.30:
        return base
    if r < 0.55:
        return base + str(random.randint(0, 99))
    if r < 0.75:
        return base.capitalize() + random.choice(["", "1", "123", "!", "@123"])
    return base + str(random.randint(1950, 2025))


def gen_short_random():
    pool = random.choice([string.ascii_lowercase, string.ascii_lowercase + string.digits,
                          string.ascii_letters + string.digits])
    return _rand_chars(random.randint(4, 7), pool)


def gen_word_digits():
    w = _decorate_case(random.choice(_words()))
    tail = random.choice([str(random.randint(0, 99)), str(random.randint(100, 9999)),
                          str(random.randint(1970, 2025))])
    sym = random.choice(["", "", random.choice("!@#$")])
    return w + tail + sym


def gen_leet_word():
    w = _decorate_case(_leetify(random.choice(_words())))
    return w + random.choice(["", "1", "123", "!", "2024", "@" + str(random.randint(10, 99))])


def gen_two_words():
    a, b = random.sample(_words(), 2)
    sep = random.choice(["", "", "", "-", "_", ".", str(random.randint(0, 99))])
    return _decorate_case(a) + sep + _decorate_case(b) + random.choice(["", "", str(random.randint(0, 99))])


def gen_passphrase():
    k = random.randint(3, 5)
    sep = random.choice(["-", " ", "_", ".", "", "!"])
    words = [_decorate_case(w) for w in random.sample(_words(), k)]
    phrase = sep.join(words)
    if random.random() < 0.5:
        phrase += random.choice(string.digits) + random.choice(SYMBOLS)
    return phrase


def gen_random_alnum():
    return _rand_chars(random.randint(8, 14), string.ascii_letters + string.digits)


def gen_random_lower_digits():
    return _rand_chars(random.randint(8, 14), string.ascii_lowercase + string.digits)


def gen_random_full():
    return _rand_chars(random.randint(10, 24), string.ascii_letters + string.digits + SYMBOLS)


def gen_pattern():
    p = random.choice(KEYBOARD_PATTERNS)
    r = random.random()
    if r < 0.4:
        return p
    if r < 0.7:
        return p + str(random.randint(0, 999))
    return p.capitalize() + random.choice(["!", "@", "#"]) + str(random.randint(0, 99))


def gen_repeats():
    ch = random.choice(string.ascii_letters + string.digits)
    n = random.randint(5, 14)
    s = ch * n
    if random.random() < 0.5:
        s += random.choice(string.digits) * random.randint(1, 3)
    return s


def gen_digits_only():
    return _rand_chars(random.randint(4, 14), string.digits)


def gen_name_pattern():
    w = _decorate_case(random.choice(_words()))
    return w + random.choice(["@", "#", "_", "."]) + str(random.randint(1, 9999))


def gen_mixed_medium():
    # random-ish but short / low-variety: typical "medium" passwords
    pool = random.choice([string.ascii_lowercase + string.digits,
                          string.ascii_letters,
                          string.ascii_letters + string.digits])
    return _rand_chars(random.randint(7, 10), pool)


def gen_strong_phrase_mix():
    w1, w2 = random.sample(EXTRA_WORDS, 2)
    return (w1.capitalize() + random.choice(SYMBOLS) + _rand_chars(3, string.digits)
            + random.choice(SYMBOLS) + w2.capitalize() + _rand_chars(2, string.ascii_letters))


GENERATORS = [
    (gen_common_variant, 10), (gen_short_random, 6), (gen_word_digits, 12),
    (gen_leet_word, 6), (gen_two_words, 8), (gen_passphrase, 6),
    (gen_random_alnum, 9), (gen_random_lower_digits, 6), (gen_random_full, 9),
    (gen_pattern, 4), (gen_repeats, 3), (gen_digits_only, 4),
    (gen_name_pattern, 6), (gen_mixed_medium, 8), (gen_strong_phrase_mix, 3),
]


def build_dataset(size, seed=42):
    random.seed(seed)
    funcs, weights = zip(*GENERATORS)
    seen = set()
    rows = []
    attempts = 0
    # generate until `size` UNIQUE passwords exist (de-duplication, as in the slides)
    while len(rows) < size and attempts < size * 20:
        attempts += 1
        pw = random.choices(funcs, weights=weights, k=1)[0]()
        if not pw or pw in seen:
            continue
        seen.add(pw)
        rows.append((pw, policy_label(pw)))
    return pd.DataFrame(rows, columns=["password", "strength"])


def main():
    parser = argparse.ArgumentParser(description="Generate a labelled password dataset")
    parser.add_argument("--size", type=int, default=60000, help="number of unique passwords")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "password_dataset.csv"))
    args = parser.parse_args()

    df = build_dataset(args.size, args.seed)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} unique labelled passwords to {args.out}")
    print(df["strength"].value_counts().to_string())


if __name__ == "__main__":
    main()
