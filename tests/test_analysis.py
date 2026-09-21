"""Tests for the tokenizer, stop-words and porter-lite stemmer."""
import unittest

from minisearch.analysis import TOKEN_RE, analyze, analyze_with_positions, stem, tokenize_raw


class TestTokenizeRaw(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(tokenize_raw("Hello WORLD"), ["hello", "world"])

    def test_splits_punctuation(self):
        self.assertEqual(tokenize_raw("hello, world!"), ["hello", "world"])

    def test_unicode_words(self):
        toks = tokenize_raw("caf\u00e9 na\u00efve")
        self.assertEqual(toks, ["caf\u00e9", "na\u00efve"])

    def test_digits_kept(self):
        self.assertEqual(tokenize_raw("python3 rocks 42"), ["python", "3", "rocks", "42"])

    def test_empty(self):
        self.assertEqual(tokenize_raw(""), [])

    def test_underscores_split(self):
        self.assertEqual(tokenize_raw("foo_bar"), ["foo", "bar"])


class TestStopwords(unittest.TestCase):
    def test_common_words_removed(self):
        self.assertEqual(analyze("the quick brown fox"), ["quick", "brown", "fox"])

    def test_all_stopwords(self):
        self.assertEqual(analyze("the and of to"), [])

    def test_stopword_positions_leave_no_gap(self):
        got = analyze_with_positions("quick the brown")
        self.assertEqual(got, [("quick", 0), ("brown", 1)])


class TestStemmer(unittest.TestCase):
    def test_plural(self):
        self.assertEqual(stem("cats"), "cat")

    def test_boxes(self):
        self.assertEqual(stem("boxes"), "box")

    def test_stories(self):
        self.assertEqual(stem("stories"), "story")

    def test_running(self):
        self.assertEqual(stem("running"), "run")

    def test_connected(self):
        self.assertEqual(stem("connected"), "connect")

    def test_quickly(self):
        self.assertEqual(stem("quickly"), "quick")

    def test_short_words_untouched(self):
        self.assertEqual(stem("as"), "as")
        self.assertEqual(stem("a"), "a")

    def test_search_stems_match(self):
        self.assertEqual(analyze("running"), analyze("run"))


class TestAnalyze(unittest.TestCase):
    def test_positions_sequential(self):
        got = dict((t, p) for t, p in analyze_with_positions("alpha beta gamma"))
        self.assertEqual(got, {"alpha": 0, "beta": 1, "gamma": 2})

    def test_empty_string(self):
        self.assertEqual(analyze(""), [])
        self.assertEqual(analyze_with_positions(""), [])


if __name__ == "__main__":
    unittest.main()
