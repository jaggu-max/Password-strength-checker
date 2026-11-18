"""Phase 1 - Unit testing: feature extraction functions (see project slides)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.feature_extraction import (  # noqa: E402
    FEATURE_NAMES, build_suggestions, extract_features, extract_vector,
    max_repeat_run, max_sequence_len, shannon_entropy,
)


class TestFeatures(unittest.TestCase):
    def test_basic_counts(self):
        f = extract_features("Ab1!xyz9")
        self.assertEqual(f["length"], 8)
        self.assertEqual(f["n_upper"], 1)
        self.assertEqual(f["n_lower"], 4)
        self.assertEqual(f["n_digits"], 2)
        self.assertEqual(f["n_symbols"], 1)
        self.assertEqual(f["n_classes"], 4)

    def test_entropy(self):
        self.assertEqual(shannon_entropy(""), 0.0)
        self.assertEqual(shannon_entropy("aaaa"), 0.0)
        self.assertAlmostEqual(shannon_entropy("abcd"), 2.0)       # 4 equal symbols -> 2 bits
        self.assertAlmostEqual(shannon_entropy("aabb"), 1.0)

    def test_total_entropy_uses_charset(self):
        f = extract_features("abcdefgh")                            # lowercase only -> pool 26
        self.assertAlmostEqual(f["total_entropy_bits"], 8 * math.log2(26))

    def test_repeats_and_sequences(self):
        self.assertEqual(max_repeat_run("aaabb"), 3)
        self.assertGreaterEqual(max_sequence_len("xabcx"), 3)
        self.assertGreaterEqual(max_sequence_len("x321x"), 3)
        self.assertGreaterEqual(max_sequence_len("myqwerty"), 3)
        self.assertEqual(max_sequence_len("a8Kd92Lm"), 0)

    def test_dictionary_and_common(self):
        self.assertEqual(extract_features("password")["is_common_password"], 1)
        self.assertEqual(extract_features("P@ssw0rd!")["has_dictionary_word"], 1)  # leet-speak aware
        self.assertEqual(extract_features("xK9#mQ2$vL7@pR4!")["has_dictionary_word"], 0)

    def test_vector_shape_matches_feature_names(self):
        vec = extract_vector("Hello123!")
        self.assertEqual(vec.shape, (1, len(FEATURE_NAMES)))

    def test_empty_and_unicode(self):
        self.assertEqual(extract_features("")["length"], 0)       # must not crash
        self.assertEqual(extract_features("pässwörd✓")["length"], 9)

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            extract_features(12345)

    def test_suggestions(self):
        tips = " ".join(build_suggestions("abc"))
        self.assertIn("12 characters", tips)
        self.assertIn("uppercase", tips)
        self.assertIn("special", tips)


if __name__ == "__main__":
    unittest.main()
