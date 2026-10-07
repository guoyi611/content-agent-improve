import os
import tempfile
import unittest
from pathlib import Path

from run import save_article


class RunOutputTests(unittest.TestCase):
    def test_empty_article_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                self.assertIsNone(save_article("", "xiaohongshu", timestamp=123))
                self.assertEqual(list(Path(directory).glob("output_*.md")), [])
            finally:
                os.chdir(previous)

    def test_non_empty_article_is_saved_with_trailing_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                filename = save_article("文章正文", "xiaohongshu", timestamp=123)
                self.assertEqual(filename, "output_xiaohongshu_123.md")
                self.assertEqual(
                    Path(filename).read_text(encoding="utf-8"),
                    "文章正文\n",
                )
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
