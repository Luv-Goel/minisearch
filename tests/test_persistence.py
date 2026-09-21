"""Persistence: commit, reopen, crash recovery, tombstones."""
import json
import os
import tempfile
import unittest

from minisearch import Index


class TestPersistence(unittest.TestCase):
    def test_reopen_after_commit(self):
        tmp = tempfile.mkdtemp()
        idx = Index.create(tmp)
        idx.add_document("1", {"title": "hello", "body": "hello world"})
        idx.commit()
        idx2 = Index.open(tmp)
        hits = idx2.search("hello")
        self.assertEqual([h.doc_id for h in hits], ["1"])
        self.assertEqual(idx2.get("1"), {"title": "hello", "body": "hello world"})

    def test_uncommitted_changes_not_visible_after_reopen(self):
        tmp = tempfile.mkdtemp()
        idx = Index.create(tmp)
        idx.add_document("1", {"body": "committed doc"})
        idx.commit()
        idx.add_document("2", {"body": "uncommitted doc"})
        idx2 = Index.open(tmp)
        self.assertEqual(sorted(idx2.doc_ids()), ["1"])

    def test_delete_survives_reopen(self):
        tmp = tempfile.mkdtemp()
        idx = Index.create(tmp)
        idx.add_document("1", {"body": "python tutorial"})
        idx.add_document("2", {"body": "python recipes"})
        idx.delete_document("1")
        idx.commit()
        idx2 = Index.open(tmp)
        hits = idx2.search("python")
        self.assertEqual([h.doc_id for h in hits], ["2"])

    def test_tmp_orphans_ignored_on_open(self):
        tmp = tempfile.mkdtemp()
        idx = Index.create(tmp)
        idx.add_document("1", {"body": "hello world"})
        idx.commit()
        with open(os.path.join(tmp, ".seg_99999.json.tmp"), "w") as f:
            f.write("{garbage")
        idx2 = Index.open(tmp)
        self.assertEqual([h.doc_id for h in idx2.search("hello")], ["1"])

    def test_manifest_written_and_seq_increments(self):
        tmp = tempfile.mkdtemp()
        idx = Index.create(tmp)
        s1 = idx.commit()
        idx.add_document("1", {"body": "hello"})
        s2 = idx.commit()
        self.assertGreater(s2, s1)
        with open(os.path.join(tmp, "manifest.json")) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["seq"], s2)
        self.assertEqual(len(manifest["segments"]), 1)

    def test_stats(self):
        tmp = tempfile.mkdtemp()
        idx = Index.create(tmp)
        idx.add_document("1", {"body": "hello world"})
        st = idx.stats()
        self.assertEqual(st["num_docs"], 1)
        self.assertGreater(st["num_terms"], 0)


if __name__ == "__main__":
    unittest.main()
