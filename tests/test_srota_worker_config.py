import unittest

from scripts.srota_asr_worker import configured_cpu_threads


class SrotaWorkerConfigTests(unittest.TestCase):
    def test_default_is_four_threads(self):
        self.assertEqual(configured_cpu_threads({}), 4)

    def test_environment_override_is_respected(self):
        self.assertEqual(
            configured_cpu_threads({"SPEECHMIND_SROTA_TORCH_THREADS": "6"}),
            6,
        )

    def test_thread_override_is_bounded(self):
        self.assertEqual(
            configured_cpu_threads({"SPEECHMIND_SROTA_TORCH_THREADS": "100"}),
            16,
        )
        self.assertEqual(
            configured_cpu_threads({"SPEECHMIND_SROTA_TORCH_THREADS": "-2"}),
            1,
        )

    def test_invalid_override_uses_default(self):
        self.assertEqual(
            configured_cpu_threads({"SPEECHMIND_SROTA_TORCH_THREADS": "fast"}),
            4,
        )


if __name__ == "__main__":
    unittest.main()
