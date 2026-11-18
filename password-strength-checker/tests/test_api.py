"""Phase 2 - Integration testing: Flask routes trigger ML predictions (200 OK)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, BUNDLE  # noqa: E402


@unittest.skipIf(BUNDLE is None, "model.pkl missing - run python ml/train_model.py first")
class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.login()

    def post(self, payload):
        return self.client.post("/api/check", json=payload)

    def login(self, name="Test User", password="Admin@123"):
        return self.client.post("/login", data={"name": name, "password": password})

    def test_home_page(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").get_json()["status"], "ok")

    def test_weak(self):
        r = self.post({"password": "password"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["category"], "weak")

    def test_strong(self):
        r = self.post({"password": "xK9#mQ2$vL7@pR4!zT"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["category"], "strong")

    def test_response_shape(self):
        d = self.post({"password": "Hello123!"}).get_json()
        for key in ("category", "confidence", "probabilities", "suggestions", "analysis", "model"):
            self.assertIn(key, d)
        self.assertIn(d["category"], ("weak", "medium", "strong"))
        self.assertTrue(0.0 <= d["confidence"] <= 1.0)

    def test_empty_password_rejected(self):
        self.assertEqual(self.post({"password": ""}).status_code, 400)

    def test_missing_field_rejected(self):
        self.assertEqual(self.post({"pwd": "abc"}).status_code, 400)

    def test_wrong_type_rejected(self):
        self.assertEqual(self.post({"password": 12345}).status_code, 400)

    def test_too_long_rejected(self):
        self.assertEqual(self.post({"password": "a" * 129}).status_code, 400)

    def test_non_json_rejected(self):
        r = self.client.post("/api/check", data="password=abc", content_type="text/plain")
        self.assertEqual(r.status_code, 415)

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get("/api/check").status_code, 405)

    def test_no_store_header(self):
        r = self.post({"password": "abc"})
        self.assertEqual(r.headers.get("Cache-Control"), "no-store")

    def test_password_not_echoed(self):
        secret = "UniqueSecret_#98127"
        r = self.post({"password": secret})
        self.assertNotIn(secret, r.get_data(as_text=True))
        bad = self.post({"password": "x" * 500 + secret})
        self.assertNotIn(secret, bad.get_data(as_text=True))

    def test_model_info(self):
        r = self.client.get("/api/model-info")
        self.assertEqual(r.status_code, 200)
        self.assertIn("best_model", r.get_json())


class TestAuthentication(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_shows_login_when_signed_out(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome back", response.data)

    def test_login_rejects_incorrect_password(self):
        response = self.client.post(
            "/login", data={"name": "Mohan", "password": "incorrect"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Incorrect password", response.data)
        self.assertEqual(self.client.get("/welcome").status_code, 302)

    def test_login_requires_name(self):
        response = self.client.post(
            "/login", data={"name": " ", "password": "Admin@123"}
        )
        self.assertIn(b"Please enter your name", response.data)

    def test_login_welcome_dashboard_and_logout(self):
        response = self.client.post(
            "/login", data={"name": "Mohan", "password": "Admin@123"}
        )
        self.assertEqual(response.status_code, 302)
        welcome = self.client.get("/welcome")
        self.assertIn(b"Welcome, Mohan!", welcome.data)
        dashboard = self.client.get("/")
        self.assertIn(b"Welcome, Mohan / WORKSPACE", dashboard.data)
        self.assertEqual(self.client.post("/logout").status_code, 302)
        self.assertIn(b"Welcome back", self.client.get("/").data)

    def test_password_api_requires_login(self):
        response = self.client.post("/api/check", json={"password": "abc"})
        self.assertEqual(response.status_code, 401)

    def test_password_checker_api_request_keeps_session_credentials(self):
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static", "js", "app.js",
        )
        with open(script_path, encoding="utf-8") as script_file:
            script = script_file.read()
        self.assertIn('credentials: "same-origin"', script)
        self.assertNotIn('credentials: "omit"', script)


if __name__ == "__main__":
    unittest.main()
