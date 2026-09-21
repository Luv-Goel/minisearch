"""HTTP round-trip tests (stdlib server + urllib client)."""
import tempfile
import threading
import unittest

from minisearch import client as client_mod
from minisearch.server import INDEX_REGISTRY, build_server


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        INDEX_REGISTRY.clear()
        cls.server = build_server("127.0.0.1", 0, cls.tmp)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()
        cls.client = client_mod.MiniSearchClient(
            f"http://127.0.0.1:{cls.port}")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_health(self):
        self.assertEqual(self.client.health()["ok"], True)

    def test_put_search_delete_roundtrip(self):
        self.client.put_doc("books", "1",
                            {"title": "python guide", "body": "search engine"})
        self.client.put_doc("books", "2",
                            {"title": "cooking", "body": "pasta recipes"})
        res = self.client.search("books", "python")
        self.assertEqual([h["id"] for h in res["hits"]], ["1"])
        self.assertGreater(res["hits"][0]["score"], 0)
        self.assertIn("snippet", res["hits"][0])
        self.client.delete_doc("books", "1")
        res = self.client.search("books", "python")
        self.assertEqual(res["hits"], [])

    def test_stats_and_commit(self):
        self.client.put_doc("wiki", "1", {"body": "hello world"})
        stats = self.client.stats("wiki")
        self.assertGreaterEqual(stats["num_docs"], 1)

    def test_search_missing_query_is_400(self):
        import urllib.request, urllib.error, json
        url = f"http://127.0.0.1:{self.port}/indexes/books/search"
        req = urllib.request.Request(
            url, data=json.dumps({}).encode(), method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_route_is_404(self):
        with self.assertRaises(client_mod.MiniSearchError) as ctx:
            self.client._request("GET", "/nope")
        self.assertEqual(ctx.exception.status, 404)


if __name__ == "__main__":
    unittest.main()
