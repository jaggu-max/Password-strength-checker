"""
Flask backend for the ML-Based Password Strength Checker.

Routes
  GET  /                 the web page
  POST /api/check        JSON {"password": "..."}  ->  predicted category
  GET  /api/model-info   training metrics (read from models/metrics.json)
  GET  /api/health       simple status check

PRIVACY: passwords are processed in memory only. They are never written to
disk, never logged, and never included in error messages.
"""

import hmac
import json
import logging
import os
from functools import wraps

import joblib
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from ml.feature_extraction import (
    FEATURE_NAMES,
    build_suggestions,
    extract_features,
    features_to_vector,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

MAX_PASSWORD_LENGTH = 128

app = Flask(__name__)
app.secret_key = os.environ.get("PASSWORD_CHECKER_SECRET_KEY") or os.urandom(32)
LOGIN_PASSWORD = os.environ.get("PASSWORD_CHECKER_LOGIN_PASSWORD", "Admin@123")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_HTTPS", "").lower() == "true",
)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024  # request bodies above 4 KB are rejected

# Werkzeug logs only the request line ("POST /api/check ..."), never the body.
# We keep it at INFO and never log anything derived from the password ourselves.
logging.getLogger("werkzeug").setLevel(logging.INFO)


# ------------------------------------------------------------ model loading
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    bundle = joblib.load(MODEL_PATH)
    # Integration check from the slides: feature vector shape must match training.
    if list(bundle["feature_names"]) != list(FEATURE_NAMES):
        raise RuntimeError(
            "models/model.pkl was trained with different features. "
            "Run: python ml/train_model.py"
        )
    return bundle


try:
    BUNDLE = load_model()
except Exception as exc:  # keep the site up and explain the problem
    BUNDLE = None
    print(f"[startup] Could not load model: {exc}")

if BUNDLE is None:
    print("[startup] No trained model found. Run: python ml/generate_dataset.py "
          "&& python ml/train_model.py")


# ------------------------------------------------------------------ helpers
def error(message, status):
    return jsonify({"error": message}), status


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_name"):
            if request.path.startswith("/api/"):
                return error("Please log in to continue.", 401)
            return redirect(url_for("index"))
        return view(*args, **kwargs)
    return wrapped


@app.after_request
def add_headers(resp):
    # Responses can contain feedback about a password -> never cache them.
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    if not session.get("user_name"):
        return render_template("login.html", error_message=None, name="")
    return render_template("index.html", model_ready=BUNDLE is not None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_name"):
        return redirect(url_for("welcome"))

    name = ""
    error_message = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "")
        if not name or len(name) > 80:
            error_message = "Please enter your name (up to 80 characters)."
        elif not hmac.compare_digest(password, LOGIN_PASSWORD):
            error_message = "Incorrect password. Please try again."
        else:
            session.clear()
            session["user_name"] = name
            return redirect(url_for("welcome"))

    return render_template("login.html", error_message=error_message, name=name)


@app.get("/welcome")
@login_required
def welcome():
    return render_template("welcome.html", user_name=session["user_name"])


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.get("/api/health")
@login_required
def health():
    return jsonify({
        "status": "ok" if BUNDLE else "model_missing",
        "model": BUNDLE["model_name"] if BUNDLE else None,
    })


@app.get("/api/model-info")
@login_required
def model_info():
    if not os.path.exists(METRICS_PATH):
        return error("Model metrics not found. Run python ml/train_model.py", 404)
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return jsonify(json.load(fh))


@app.post("/api/check")
@login_required
def check_password():
    if BUNDLE is None:
        return error("The ML model is not available on the server. "
                     "Run python ml/train_model.py and restart.", 503)

    if not request.is_json:
        return error("Request must be JSON with a 'password' field.", 415)

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "password" not in data:
        return error("Missing 'password' field.", 400)

    password = data["password"]
    if not isinstance(password, str):
        return error("Password must be text.", 400)
    if len(password) == 0:
        return error("Please enter a password to check.", 400)
    if len(password) > MAX_PASSWORD_LENGTH:
        return error(f"Password is too long (maximum {MAX_PASSWORD_LENGTH} characters).", 400)

    try:
        feats = extract_features(password)
        vector = features_to_vector(feats)
        model = BUNDLE["model"]
        class_names = BUNDLE["class_names"]

        pred_index = int(model.predict(vector)[0])
        probs = model.predict_proba(vector)[0]
        classes = [int(c) for c in model.classes_]
        prob_map = {class_names[c]: round(float(p), 4) for c, p in zip(classes, probs)}
    except Exception:
        # Deliberately generic: never echo the password or a traceback to the client.
        return error("Something went wrong while analysing the password.", 500)

    return jsonify({
        "category": class_names[pred_index],
        "confidence": prob_map.get(class_names[pred_index], 0.0),
        "probabilities": prob_map,
        "suggestions": build_suggestions(password, feats),
        "analysis": {
            "length": feats["length"],
            "uppercase": feats["n_upper"],
            "lowercase": feats["n_lower"],
            "digits": feats["n_digits"],
            "symbols": feats["n_symbols"],
            "repeated_chars": feats["repeated_chars"],
            "entropy_bits_per_char": round(feats["entropy_per_char"], 2),
            "charset_size": feats["charset_size"],
            "dictionary_word": bool(feats["has_dictionary_word"]),
            "common_password": bool(feats["is_common_password"]),
        },
        "model": BUNDLE["model_name"],
    })


@app.errorhandler(404)
def not_found(_e):
    return error("Not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_e):
    return error("Method not allowed.", 405)


@app.errorhandler(413)
def too_large(_e):
    return error("Request too large.", 413)


@app.errorhandler(500)
def server_error(_e):
    return error("Internal server error.", 500)


if __name__ == "__main__":
    # debug=False: the debugger can expose request data in tracebacks.
    app.run(host="127.0.0.1", port=5000, debug=False)
