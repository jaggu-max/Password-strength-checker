# Machine Learning-Based Password Strength Checker

A web application that classifies a password as **Weak**, **Medium** or **Strong** using machine learning.
Built to follow the project presentation (Mini Project Review 2: System Design and Implementation).

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python Flask (REST API)
- **Machine learning:** scikit-learn (Decision Tree, Random Forest, Logistic Regression)
- **Data processing:** Pandas, NumPy (+ imbalanced-learn for SMOTE)

---

## 1. Project structure

```
password-strength-checker/
├── app.py                      Flask application (routes, validation, model loading)
├── requirements.txt            Python dependencies
├── README.md                   This file
├── ml/
│   ├── feature_extraction.py   Feature extraction (used by BOTH training and Flask)
│   ├── generate_dataset.py     Builds the labelled dataset (data/password_dataset.csv)
│   └── train_model.py          Trains, compares and saves the models
├── data/
│   ├── password_dataset.csv    Labelled dataset (password, strength)
│   ├── common_passwords.txt    List of very common passwords
│   └── common_words.txt        Dictionary words used for pattern detection
├── models/
│   ├── model.pkl               Saved best model (loaded by Flask)
│   └── metrics.json            Measured results of all 3 models (shown on the website)
├── templates/index.html        Frontend page
├── static/css/style.css        Responsive paper-and-forest technical UI
├── static/js/app.js            Frontend logic (calls the Flask API)
├── static/img/mosaic.svg       Repeating technical grid background
└── tests/                      Unit tests + API integration tests
```

---

## 2. Installation

Requires **Python 3.9 or newer**.

**Windows (Command Prompt / PowerShell)**
```bash
cd password-strength-checker
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```bash
cd password-strength-checker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Run the application

The project already includes a generated dataset and a trained model, so you can start immediately:

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### Login

The home page opens the login screen. Enter a name and use the demo password
`Admin@123`, then continue from the welcome screen to the existing checker.
The dashboard and password-checking API require an authenticated session.

For local development, the default credential is intentionally simple. You can
override it with `PASSWORD_CHECKER_LOGIN_PASSWORD` and set a stable
`PASSWORD_CHECKER_SECRET_KEY` before starting Flask. For example, in
PowerShell:

```powershell
$env:PASSWORD_CHECKER_LOGIN_PASSWORD = "your-local-password"
$env:PASSWORD_CHECKER_SECRET_KEY = "a-long-random-secret"
python app.py
```

Set `FLASK_HTTPS=true` when serving over HTTPS to enable the secure session
cookie flag. This single shared demo password is not a production
authentication system.

### Re-train the models (optional)

```bash
python ml/generate_dataset.py      # creates data/password_dataset.csv
python ml/train_model.py           # trains 3 models, saves models/model.pkl + models/metrics.json
```

The training script prints accuracy, precision, recall, F1-score and the confusion matrix for every model,
then picks the best one automatically. Restart `python app.py` afterwards so Flask loads the new model.

### Run the tests

```bash
python -m unittest discover -s tests -v
```

---

## 4. Using the real Kaggle dataset (recommended for your report)

The presentation refers to the Kaggle *Password Strength Classifier* dataset (000webhost leak).
That file cannot be bundled with this project, so the included dataset is **generated** (see section 6).
To train on the real file:

1. Download `data.csv` from Kaggle (columns `password`, `strength` with values 0 / 1 / 2).
2. Run:
   ```bash
   python ml/train_model.py --data path/to/data.csv --sample 150000
   ```
   `--sample` uses a random subset to keep grid search fast; remove it to use every row
   (Random Forest grid search on the full file can take a long time).
3. Restart `python app.py`. The website's "Model results" section updates automatically from `models/metrics.json`.

Rows with broken CSV formatting are skipped, nulls and duplicates are removed, and labels 0/1/2 are mapped to weak/medium/strong.

---

## 5. How it works (matches the slides)

1. **User input** – the password is typed in the web page.
2. **Transmission** – JavaScript sends it as JSON to `POST /api/check`.
3. **Processing** – `ml/feature_extraction.py` converts it into 21 numerical features:
   length, uppercase / lowercase / digit / symbol counts, number of character classes,
   unique-character ratio, repeated characters, longest repeat run, sequences (abc, 123, qwerty),
   Shannon entropy, character-set size, total entropy, dictionary-word hit (leet-speak aware),
   common-password flag, leet count, trailing digits and character ratios.
4. **Prediction** – the saved model in `models/model.pkl` predicts weak / medium / strong.
5. **Feedback** – the page shows the category, a strength indicator, analysis and improvement tips.

### Training pipeline (`ml/train_model.py`)

| Step | What happens |
|---|---|
| Cleaning | null removal, de-duplication, label mapping |
| Split | stratified **70% train / 10% validation / 20% test** |
| Balancing | **SMOTE on training data only** (inside the cross-validation pipeline; never applied to validation/test or to live scoring) |
| Tuning | `GridSearchCV`, 3-fold cross-validation, scored on weighted F1 |
| Models | Decision Tree, Random Forest, Logistic Regression (with feature scaling) |
| Selection | best **validation weighted F1** (accuracy breaks ties) |
| Reporting | accuracy, precision, recall, F1, confusion matrix on the test set |

### API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Login page when signed out; password checker dashboard when signed in |
| GET / POST | `/login` | Sign in with a name and the configured demo password |
| GET | `/welcome` | Personalized welcome screen |
| POST | `/logout` | End the current session |
| POST | `/api/check` | Authenticated JSON body `{"password": "..."}` → category, class probability, suggestions, analysis |
| GET | `/api/model-info` | Authenticated measured training results (from `models/metrics.json`) |
| GET | `/api/health` | Authenticated server / model status |

Errors return JSON with a clear message: `400` (empty, wrong type, too long – max 128 chars),
`405`, `413` (body too large), `415` (not JSON), `503` (model missing), `500` (generic).

---

## 6. Important notes (please read before presenting)

- **The bundled dataset is generated, not the Kaggle data.** Its labels come from a documented scoring policy
  in `ml/generate_dataset.py`. Because the labels are rule-based, the models score very high; this shows they
  have learned the policy, **not** that they predict real-world cracking resistance. All numbers shown on the
  website come from `models/metrics.json`, written by the real training run. None are hard-coded.
- **The best model is chosen by measurement.** On the bundled dataset this is not necessarily Random Forest,
  even though the slides name it the champion model. Check `models/metrics.json` after training and describe
  your actual result in the report. With the Kaggle dataset the winner may differ.
- **TF-IDF character n-grams** (mentioned on the feature-extraction slide) are **not** used. The model uses
  21 numerical features, which keeps the model small and the live prediction fast. They can be added later.

---

## 7. Security and privacy

- Passwords are processed **in memory only**: never written to disk, never logged, never echoed back in responses or errors.
- Flask runs with `debug=False`; error responses are generic and contain no tracebacks.
- API responses are sent with `Cache-Control: no-store`; the browser request uses `cache: "no-store"` and omits cookies.
- Checker passwords are sent only to the authenticated API, are not stored in cookies or localStorage, and are not included in session data. Same-origin requests include the signed, HTTP-only session cookie; no analytics are used.
- Request bodies above 4 KB are rejected.
- This is a learning project. For real systems, send passwords only over HTTPS, and never reuse the password you test here.

---

## 8. Testing phases (from the slides)

| Phase | Where |
|---|---|
| 1. Unit testing – feature extraction (e.g. entropy) | `tests/test_features.py` |
| 2. Integration testing – Flask routes → ML prediction, `200 OK` | `tests/test_api.py` |
| 3. User acceptance testing – UI responsiveness, clarity of suggestions | Try it in the browser on desktop and mobile |
