import unittest

from utils.translation_cache import TranslationResultCache


class TranslationResultCacheTests(unittest.TestCase):
    def test_cache_hits_are_language_and_text_specific(self):
        cache = TranslationResultCache(max_items=2)
        cache.put("नमस्ते", "hin_Deva", "Hello")
        self.assertEqual(cache.get("नमस्ते", "hin_Deva"), "Hello")
        self.assertIsNone(cache.get("नमस्ते", "ben_Beng"))
        self.assertIsNone(cache.get("नमस्कार", "hin_Deva"))

    def test_cache_evicts_least_recently_used_result(self):
        cache = TranslationResultCache(max_items=2)
        cache.put("a", "hi", "A")
        cache.put("b", "hi", "B")
        self.assertEqual(cache.get("a", "hi"), "A")
        cache.put("c", "hi", "C")
        self.assertIsNone(cache.get("b", "hi"))
        self.assertEqual(cache.get("a", "hi"), "A")
        self.assertEqual(cache.get("c", "hi"), "C")

    def test_empty_results_are_not_cached(self):
        cache = TranslationResultCache()
        cache.put("", "hi", "hello")
        cache.put("नमस्ते", "hi", "")
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
