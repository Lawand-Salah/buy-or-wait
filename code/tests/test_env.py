import os
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
from evidence.env import load_dotenv


class TestLoadDotenv(unittest.TestCase):
    def setUp(self):
        self.tmp_path = Path(self.id().split(".")[-1] + "_test.env")
        self.addCleanup(self.tmp_path.unlink, True)
        self.addCleanup(os.environ.pop, "TEST_LOAD_DOTENV_VAR", None)
        self.addCleanup(os.environ.pop, "TEST_LOAD_DOTENV_QUOTED", None)

    def test_parses_key_value_lines(self):
        self.tmp_path.write_text('TEST_LOAD_DOTENV_VAR=hello\n# a comment\n\nTEST_LOAD_DOTENV_QUOTED="quoted value"\n')
        load_dotenv(self.tmp_path)
        self.assertEqual(os.environ.get("TEST_LOAD_DOTENV_VAR"), "hello")
        self.assertEqual(os.environ.get("TEST_LOAD_DOTENV_QUOTED"), "quoted value")

    def test_never_overrides_an_existing_env_var(self):
        os.environ["TEST_LOAD_DOTENV_VAR"] = "already-set"
        self.tmp_path.write_text("TEST_LOAD_DOTENV_VAR=from-file\n")
        load_dotenv(self.tmp_path)
        self.assertEqual(os.environ["TEST_LOAD_DOTENV_VAR"], "already-set")

    def test_missing_file_is_a_no_op(self):
        load_dotenv(Path("definitely_does_not_exist_12345.env"))  # must not raise

    def test_malformed_lines_are_skipped_not_raised(self):
        self.tmp_path.write_text("this line has no equals sign\nTEST_LOAD_DOTENV_VAR=ok\n")
        load_dotenv(self.tmp_path)  # must not raise
        self.assertEqual(os.environ.get("TEST_LOAD_DOTENV_VAR"), "ok")


if __name__ == "__main__":
    unittest.main()
