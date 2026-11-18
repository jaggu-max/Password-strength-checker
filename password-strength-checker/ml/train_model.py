"""
Train, compare and save the password strength classifier.

Pipeline (matches the project slides):
  1. Load dataset                 (data/password_dataset.csv or your own CSV)
  2. Pre-process                  (drop nulls, remove duplicates, UTF-8 strings)
  3. Feature extraction           (ml/feature_extraction.py)
  4. Stratified split             70% train / 10% validation / 20% test
  5. SMOTE class balancing        TRAINING DATA ONLY (never validation/test)
  6. Grid search + 3-fold CV      Decision Tree, Random Forest, Logistic Regression
  7. Evaluate on validation+test  accuracy, precision, recall, F1, confusion matrix
  8. Select best model            highest validation weighted-F1 (tie -> accuracy)
  9. Save                         models/model.pkl and models/metrics.json

Usage:
    python ml/train_model.py
    python ml/train_model.py --data path/to/kaggle_data.csv --sample 150000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.feature_extraction import BASE_DIR, DATA_DIR, FEATURE_NAMES, extract_features  # noqa: E402

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:  # SMOTE is optional; training still works without it
    HAS_IMBLEARN = False

MODEL_DIR = os.path.join(BASE_DIR, "models")
CLASS_NAMES = ["weak", "medium", "strong"]  # index 0 / 1 / 2
RANDOM_STATE = 42


# ----------------------------------------------------------------- loading
def load_dataset(path, sample=None):
    """Load + clean a password dataset. Returns (DataFrame, cleaning_report)."""
    # on_bad_lines='skip' handles the Kaggle file, where some passwords contain commas
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", dtype=str,
                     keep_default_na=False, na_values=[""])
    report = {"rows_loaded": int(len(df))}

    if not {"password", "strength"}.issubset(df.columns):
        raise ValueError("CSV must contain the columns 'password' and 'strength'")
    df = df[["password", "strength"]].dropna()
    report["rows_after_null_removal"] = int(len(df))

    # accept 0/1/2 or weak/medium/strong
    mapping = {"0": 0, "1": 1, "2": 2, "weak": 0, "medium": 1, "strong": 2}
    df["label"] = df["strength"].str.strip().str.lower().map(mapping)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df.drop_duplicates(subset="password")
    report["rows_after_deduplication"] = int(len(df))

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=RANDOM_STATE)
        report["rows_sampled"] = int(sample)
    return df.reset_index(drop=True), report


def build_feature_matrix(passwords):
    rows = [list(extract_features(p).values()) for p in passwords]
    return np.array(rows, dtype=float)


# ------------------------------------------------------------------ models
def model_definitions():
    """name -> (estimator, param_grid, needs_scaling)"""
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [8, 12, 20, None], "min_samples_leaf": [1, 5]},
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [15, None], "min_samples_leaf": [1, 3]},
            False,
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {"C": [0.1, 1.0, 10.0]},
            True,
        ),
    }


def make_search_pipeline(estimator, needs_scaling, use_smote):
    """SMOTE sits INSIDE the CV pipeline so it only ever touches training folds."""
    steps = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps) if use_smote else SkPipeline(steps)


def deployable_pipeline(fitted):
    """Drop the SMOTE step: training-only steps are NOT part of the online scoring path."""
    keep = [(n, s) for n, s in fitted.steps if n != "smote"]
    return SkPipeline(keep)


def evaluate(model, X, y):
    pred = model.predict(X)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    report = classification_report(y, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "per_class": {
            c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
            for c in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train password strength models")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "password_dataset.csv"))
    ap.add_argument("--sample", type=int, default=None,
                    help="randomly use only N rows (useful for the 669k Kaggle file)")
    ap.add_argument("--no-smote", action="store_true", help="skip SMOTE class balancing")
    ap.add_argument("--cv", type=int, default=3, help="cross-validation folds")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"Dataset not found: {args.data}\nRun: python ml/generate_dataset.py")

    use_smote = HAS_IMBLEARN and not args.no_smote
    if not HAS_IMBLEARN and not args.no_smote:
        print("WARNING: imbalanced-learn is not installed -> training WITHOUT SMOTE.")

    # 1-2. load + clean
    print(f"Loading dataset: {args.data}")
    df, clean_report = load_dataset(args.data, args.sample)
    print(f"  {len(df):,} unique labelled passwords after cleaning")

    # 3. features
    print("Extracting features ...")
    t0 = time.time()
    X = build_feature_matrix(df["password"])
    y = df["label"].to_numpy()
    print(f"  feature matrix {X.shape} in {time.time() - t0:.1f}s")

    # 4. stratified 70 / 10 / 20 split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=RANDOM_STATE)  # 0.125 * 0.8 = 10%
    print(f"  split -> train {len(y_train):,} | validation {len(y_val):,} | test {len(y_test):,}")

    class_counts_train = {CLASS_NAMES[i]: int((y_train == i).sum()) for i in range(3)}
    class_counts_all = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)}

    # 5-7. grid search, evaluate
    results = {}
    fitted_models = {}
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    for name, (estimator, grid, scale) in model_definitions().items():
        print(f"\nTraining {name} (grid search, {args.cv}-fold CV) ...")
        pipe = make_search_pipeline(estimator, scale, use_smote)
        search = GridSearchCV(pipe, {f"clf__{k}": v for k, v in grid.items()},
                              scoring="f1_weighted", cv=cv, n_jobs=1, refit=True)
        t0 = time.time()
        search.fit(X_train, y_train)
        secs = time.time() - t0

        model = deployable_pipeline(search.best_estimator_)
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)
        results[name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1_weighted": float(search.best_score_),
            "train_seconds": round(secs, 1),
            "validation": val_metrics,
            "test": test_metrics,
        }
        fitted_models[name] = model
        print(f"  best params : {results[name]['best_params']}")
        print(f"  CV F1       : {search.best_score_:.4f}")
        print(f"  validation  : acc {val_metrics['accuracy']:.4f}  F1 {val_metrics['f1_weighted']:.4f}")
        print(f"  test        : acc {test_metrics['accuracy']:.4f}  F1 {test_metrics['f1_weighted']:.4f}")

    # 8. choose best model on VALIDATION weighted F1 (accuracy breaks ties)
    best_name = max(results, key=lambda n: (round(results[n]["validation"]["f1_weighted"], 6),
                                            results[n]["validation"]["accuracy"]))
    print(f"\nBest model (by validation weighted F1): {best_name}")

    # feature importances for tree models (shown on the website)
    importances = None
    final_clf = fitted_models[best_name].named_steps["clf"]
    if hasattr(final_clf, "feature_importances_"):
        pairs = sorted(zip(FEATURE_NAMES, final_clf.feature_importances_), key=lambda x: -x[1])
        importances = [{"feature": f, "importance": float(v)} for f, v in pairs[:10]]

    # 9. save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": fitted_models[best_name],
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, os.path.join(MODEL_DIR, "model.pkl"), compress=3)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "file": os.path.basename(args.data),
            "synthetic_generated": os.path.abspath(args.data) == os.path.abspath(
                os.path.join(DATA_DIR, "password_dataset.csv")),
            "cleaning": clean_report,
            "total_records_used": int(len(df)),
            "class_counts": class_counts_all,
            "train_class_counts_before_smote": class_counts_train,
            "n_features": len(FEATURE_NAMES),
        },
        "split": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test)),
                  "strategy": "stratified 70/10/20"},
        "smote_applied_to_training_only": bool(use_smote),
        "selection_rule": "highest validation weighted F1-score (accuracy breaks ties)",
        "best_model": best_name,
        "class_names": CLASS_NAMES,
        "models": results,
        "feature_importances": importances,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved models/model.pkl and models/metrics.json")


if __name__ == "__main__":
    main()
