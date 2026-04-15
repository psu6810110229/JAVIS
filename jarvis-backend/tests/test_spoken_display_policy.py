from __future__ import annotations

import unittest

from app.brain.spoken_display_policy import SpokenDisplayPolicy


class SpokenDisplayPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SpokenDisplayPolicy(max_spoken_chars=320)

    def test_redacts_raw_url_and_marks_transform(self) -> None:
        spoken, markers = self.policy.to_spoken_with_meta(
            "Check this link: https://example.com/docs?id=42"
        )
        self.assertIn("link omitted", spoken)
        self.assertNotIn("https://example.com", spoken)
        self.assertIn("redacted_url", markers)

    def test_strips_markdown_link_url_and_keeps_label(self) -> None:
        spoken, markers = self.policy.to_spoken_with_meta(
            "Read [release notes](https://example.com/release-notes) now."
        )
        self.assertIn("release notes", spoken)
        self.assertNotIn("https://example.com", spoken)
        self.assertIn("stripped_markdown_link_url", markers)

    def test_json_only_response_rewrites_to_safe_spoken_text(self) -> None:
        spoken, markers = self.policy.to_spoken_with_meta('{"tool":"search_web","status":"ok"}')
        self.assertEqual(spoken, "I completed that step.")
        self.assertIn("json_only_rewrite", markers)


if __name__ == "__main__":
    unittest.main()
