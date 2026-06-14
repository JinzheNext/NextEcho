from __future__ import annotations

import unittest

from app import app


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["service"], "NextEcho")


if __name__ == "__main__":
    unittest.main()
