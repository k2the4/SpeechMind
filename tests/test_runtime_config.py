import unittest

from utils.runtime_config import configure_torch_runtime


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    def __init__(self, interop_error=False):
        self.intra_op = None
        self.interop = None
        self.cuda = FakeCuda()
        self.interop_error = interop_error

    def set_num_threads(self, value):
        self.intra_op = value

    def set_num_interop_threads(self, value):
        if self.interop_error:
            raise RuntimeError("parallel work already started")
        self.interop = value


class RuntimeConfigTests(unittest.TestCase):
    def test_default_threads_are_capped_by_cpu_count(self):
        torch = FakeTorch()
        result = configure_torch_runtime(torch, environ={}, cpu_count=2)
        self.assertEqual(torch.intra_op, 2)
        self.assertEqual(torch.interop, 2)
        self.assertEqual(result["device"], "cpu")

    def test_environment_override_is_respected_within_safe_limit(self):
        torch = FakeTorch()
        result = configure_torch_runtime(
            torch, environ={"SPEECHMIND_TORCH_THREADS": "6"}, cpu_count=8
        )
        self.assertEqual(result["intra_op_threads"], 6)
        self.assertEqual(torch.intra_op, 6)

    def test_invalid_override_uses_default(self):
        torch = FakeTorch()
        result = configure_torch_runtime(
            torch, environ={"SPEECHMIND_TORCH_THREADS": "not-a-number"}, cpu_count=8
        )
        self.assertEqual(result["intra_op_threads"], 4)

    def test_interop_already_initialized_does_not_break_runtime(self):
        torch = FakeTorch(interop_error=True)
        result = configure_torch_runtime(torch, environ={}, cpu_count=8)
        self.assertEqual(torch.intra_op, 4)
        self.assertTrue(result["interop_configuration_skipped"])
        self.assertIsNone(result["interop_threads"])


if __name__ == "__main__":
    unittest.main()
