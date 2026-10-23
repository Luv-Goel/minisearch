"""Search behaviour: BM25, phrase, boolean, fielded, prefix, fuzzy, delete."""
import tempfile
import threading
import unittest

from minisearch import Index


def make_index(docs):
    tmp = tempfile.mkdtemp()
    idx = Index.create(tmp)
    for doc_id, fields in docs:
        idx.add_document(doc_id, fields)
    idx.commit()
    return idx


class TestBasicSearch(unittest.TestCase):
    def test_search_returns_ranked_hits(self):
        idx = make_index([
            ("1", {"body": "python programming language"}),
            ("2", {"body": "cooking pasta recipes"}),
        ])
        hits = idx.search("python")
        self.assertEqual([h.doc_id for h in hits], ["1"])
        self.assertGreater(hits[0].score, 0)

    def test_empty_query_returns_nothing(self):
        idx = make_index([("1", {"body": "hello world"})])
        self.assertEqual(idx.search(""), [])
        self.assertEqual(idx.search("   "), [])

    def test_no_match_returns_nothing(self):
        idx = make_index([("1", {"body": "hello world"})])
        self.assertEqual(idx.search("zebra"), [])

    def test_scores_and_snippets_present(self):
        idx = make_index([("1", {"title": "hello", "body": "hello world hello"})])
        hits = idx.search("hello")
        self.assertTrue(hits[0].snippet)
        self.assertTrue(hits[0].highlights)
        self.assertIn("<em>", hits[0].highlights[0])

    def test_top_k_limits(self):
        idx = make_index([(str(i), {"body": "common word"}) for i in range(5)])
        self.assertEqual(len(idx.search("common", top_k=2)), 2)


class TestBM25(unittest.TestCase):
    def test_better_doc_ranks_higher(self):
        idx = make_index([
            ("brief", {"body": "databases store data"}),
            ("rich", {"body": "databases databases databases store data"}),
        ])
        hits = idx.search("databases")
        self.assertEqual(hits[0].doc_id, "rich")

    def test_rare_term_beats_common(self):
        idx = make_index([
            ("a", {"body": "python tutorial for python beginners"}),
            ("b", {"body": "python quantum entanglement physics"}),
        ])
        hits = idx.search("quantum physics")
        self.assertEqual(hits[0].doc_id, "b")

    def test_title_boost_changes_order(self):
        idx = make_index([
            ("a", {"title": "other words here", "body": "python python python"}),
            ("b", {"title": "python guide", "body": "other words here"}),
        ])
        plain = idx.search("python")
        boosted = idx.search("python", fields={"title": 5.0})
        self.assertEqual(boosted[0].doc_id, "b")
        self.assertGreaterEqual(len(plain), 2)


class TestPhrase(unittest.TestCase):
    def test_phrase_matches_adjacent(self):
        idx = make_index([
            ("1", {"body": "the quick brown fox jumps"}),
            ("2", {"body": "quick fox brown quick"}),
        ])
        hits = idx.search('"quick brown"')
        self.assertEqual([h.doc_id for h in hits], ["1"])

    def test_phrase_miss_on_reorder(self):
        idx = make_index([("1", {"body": "brown quick fox"})])
        self.assertEqual(idx.search('"quick brown"'), [])

    def test_fielded_phrase(self):
        idx = make_index([
            ("1", {"title": "quick brown fox", "body": "nothing relevant"}),
            ("2", {"title": "nothing relevant", "body": "quick brown fox"}),
        ])
        hits = idx.search('title:"quick brown"')
        self.assertEqual([h.doc_id for h in hits], ["1"])


class TestBoolean(unittest.TestCase):
    DOCS = [
        ("1", {"body": "python search engine tutorial"}),
        ("2", {"body": "python cooking recipes"}),
        ("3", {"body": "java search engine guide"}),
    ]

    def test_and(self):
        hits = make_index(self.DOCS).search("python AND search")
        self.assertEqual([h.doc_id for h in hits], ["1"])

    def test_or(self):
        hits = make_index(self.DOCS).search("cooking OR java")
        self.assertEqual(sorted(h.doc_id for h in hits), ["2", "3"])

    def test_not(self):
        hits = make_index(self.DOCS).search("python NOT cooking")
        self.assertEqual([h.doc_id for h in hits], ["1"])

    def test_minus_prefix(self):
        hits = make_index(self.DOCS).search("search -java")
        self.assertEqual([h.doc_id for h in hits], ["1"])

    def test_plus_prefix(self):
        hits = make_index(self.DOCS).search("+python +search")
        self.assertEqual([h.doc_id for h in hits], ["1"])

    def test_parens(self):
        hits = make_index(self.DOCS).search("(cooking OR java) AND search")
        self.assertEqual([h.doc_id for h in hits], ["3"])

    def test_implicit_or(self):
        hits = make_index(self.DOCS).search("cooking java")
        self.assertEqual(sorted(h.doc_id for h in hits), ["2", "3"])


class TestFielded(unittest.TestCase):
    def test_field_restricts(self):
        idx = make_index([
            ("1", {"title": "python guide", "body": "nothing about snakes"}),
            ("2", {"title": "cooking guide", "body": "python recipes"}),
        ])
        hits = idx.search("title:python")
        self.assertEqual([h.doc_id for h in hits], ["1"])

    def test_unknown_field_no_match(self):
        idx = make_index([("1", {"title": "python guide"})])
        self.assertEqual(idx.search("nope:python"), [])


class TestPrefixFuzzy(unittest.TestCase):
    def test_prefix(self):
        idx = make_index([
            ("1", {"body": "searching engines"}),
            ("2", {"body": "cooking recipes"}),
        ])
        self.assertEqual([h.doc_id for h in idx.search("search*")], ["1"])

    def test_fielded_prefix(self):
        idx = make_index([
            ("1", {"title": "searching engines", "body": "x"}),
            ("2", {"title": "other", "body": "searching engines"}),
        ])
        self.assertEqual([h.doc_id for h in idx.search("title:search*")],
                         ["1"])

    def test_fuzzy_distance1(self):
        idx = make_index([
            ("1", {"body": "search engine"}),
            ("2", {"body": "cooking recipes"}),
        ])
        self.assertEqual([h.doc_id for h in idx.search("serch~1")], ["1"])

    def test_fuzzy_distance2(self):
        idx = make_index([("1", {"body": "search engine"})])
        hits = idx.search("saerch~2")
        self.assertEqual([h.doc_id for h in hits], ["1"])

    def test_fuzzy_no_match(self):
        idx = make_index([("1", {"body": "search engine"})])
        self.assertEqual(idx.search("zebra~1"), [])


class TestDelete(unittest.TestCase):
    def test_delete_removes_from_results(self):
        idx = make_index([
            ("1", {"body": "python tutorial"}),
            ("2", {"body": "python recipes"}),
        ])
        self.assertTrue(idx.delete_document("1"))
        hits = idx.search("python")
        self.assertEqual([h.doc_id for h in hits], ["2"])

    def test_delete_missing_returns_false(self):
        idx = make_index([("1", {"body": "hello"})])
        self.assertFalse(idx.delete_document("nope"))

    def test_readd_after_delete(self):
        idx = make_index([("1", {"body": "python tutorial"})])
        idx.delete_document("1")
        idx.add_document("1", {"body": "completely different zebra"})
        self.assertEqual(idx.search("python"), [])
        self.assertEqual([h.doc_id for h in idx.search("zebra")], ["1"])


class TestConcurrentAdds(unittest.TestCase):
    def test_threads(self):
        tmp = tempfile.mkdtemp()
        idx = Index.create(tmp)

        def add(n):
            for j in range(10):
                idx.add_document(f"t{n}-{j}", {"body": f"worker {n} doc {j}"})

        threads = [threading.Thread(target=add, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(idx), 40)
        self.assertEqual(len(idx.search("worker", top_k=50)), 40)


class TestCount(unittest.TestCase):
    def test_count_matches_search_length(self):
        idx = make_index([
            ("1", {"body": "python programming language"}),
            ("2", {"body": "cooking pasta recipes"}),
            ("3", {"body": "python cooking basics"}),
        ])
        self.assertEqual(idx.count("python"), len(idx.search("python")))
        self.assertEqual(idx.count("zebra"), 0)
        self.assertEqual(idx.count("python OR cooking"), 3)

    def test_count_empty_query(self):
        idx = make_index([("1", {"body": "hello world"})])
        self.assertEqual(idx.count(""), 0)
        self.assertEqual(idx.count("   "), 0)

    def test_count_field_boost(self):
        idx = make_index([
            ("1", {"title": "python", "body": "java"}),
            ("2", {"title": "java", "body": "python"}),
        ])
        self.assertEqual(idx.count("python", fields={"title": 2.0}), 1)
        self.assertEqual(idx.count("python", fields={"body": 5.0}), 1)
        self.assertEqual(idx.count("python"), 2)

    def test_count_phrase(self):
        idx = make_index([
            ("1", {"body": "quick brown fox"}),
            ("2", {"body": "brown quick fox"}),
        ])
        self.assertEqual(idx.count('"quick brown"'), 1)
        self.assertEqual(idx.count("quick brown"), 2)

    def test_count_expensive_hits_equal(self):
        idx = make_index([
            ("1", {"body": "alpha beta gamma delta epsilon"}),
            ("2", {"body": "alpha beta"}),
            ("3", {"body": "gamma delta epsilon"}),
        ])
        self.assertEqual(idx.count("alpha OR gamma"), len(idx.search("alpha OR gamma", top_k=50)))


class TestDistinct(unittest.TestCase):
    def test_distinct_unique_values(self):
        idx = make_index([
            ("1", {"status": "open", "priority": "high"}),
            ("2", {"status": "closed", "priority": "low"}),
            ("3", {"status": "open", "priority": "medium"}),
        ])
        self.assertEqual(idx.distinct("status"), ["closed", "open"])
        self.assertEqual(idx.distinct("priority"), ["high", "low", "medium"])

    def test_distinct_missing_field(self):
        idx = make_index([("1", {"a": "x"})])
        self.assertEqual(idx.distinct("nonexistent"), [])

    def test_distinct_empty_index(self):
        idx = make_index([])
        self.assertEqual(idx.distinct("status"), [])

    def test_distinct_case_insensitive_field_name(self):
        idx = make_index([("1", {"Status": "open"})])
        self.assertEqual(idx.distinct("STATUS"), ["open"])


if __name__ == "__main__":
    unittest.main()
