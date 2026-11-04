from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd

from model_utils import make_mice_imputer, writable_array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit(
        "Missing required feature-selection package. Install requirements with "
        "'.venv\\Scripts\\python.exe -m pip install -r requirements.txt'. "
        f"Details: {exc}"
    )

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_PATH = RESULTS / "combined_data.csv"
RANDOM_STATE = 42
FEATURE_METHODS = [
    ("XGB-DT", 21),
    ("XGB-RF", 37),
    ("CatBoost", 71),
    ("LightGBM", 82),
]


def make_selector(name):
    """Create a tree estimator whose fitted feature_importances_ rank features."""
    if name == "XGB-DT":
        # One boosting round produces a single decision tree.
        return XGBClassifier(
            n_estimators=1, max_depth=3, learning_rate=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
        )
    if name == "XGB-RF":
        # XGBoost random-forest mode builds parallel trees with row/column sampling.
        return XGBClassifier(
            n_estimators=1, max_depth=4, random_state=RANDOM_STATE, n_jobs=1,
            learning_rate=1.0, subsample=0.8, colsample_bynode=0.8,
            objective="binary:logistic", eval_metric="logloss", verbosity=0,
            num_parallel_tree=100,
        )
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=100, depth=4, learning_rate=0.05,
            random_seed=RANDOM_STATE, verbose=False, thread_count=1,
            allow_writing_files=False,
        )
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
        )
    raise ValueError(f"Unknown feature-selection method: {name}")


def make_classifier():
    """Build a Logistic Regression pipeline with training-fold preprocessing."""
    return make_pipeline(
        FunctionTransformer(writable_array, validate=False),
        make_mice_imputer(),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def metric_values(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run src/prepare_data.py first to create results/combined_data.csv.")
    df = pd.read_csv(DATA_PATH)
    if "label" not in df:
        raise ValueError("Prepared data must contain a 'label' column. Rerun prepare_data.py.")
    y = df.pop("label").astype(int)
    metadata_names = {"patient_id", "id", "subject_id", "case_id", "sample_id"}
    feature_data = df.drop(
        columns=[c for c in df.columns if str(c).strip().lower() in metadata_names]
    )
    X = feature_data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if X.shape[1] < max(count for _, count in FEATURE_METHODS):
        raise ValueError(f"Only {X.shape[1]} numeric features are available; at least 82 are required.")
    if y.nunique() != 2:
        raise ValueError("This script expects exactly two provisional source-file classes.")
    if y.value_counts().min() < 5:
        raise ValueError("Five-fold stratified cross-validation needs at least 5 samples in each class.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))
    fold_metrics = {"All features baseline": []}
    selected_rows = []

    for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
        print(f"Processing fold {fold_number}/5...", flush=True)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        baseline = make_classifier()
        baseline.fit(X_train, y_train)
        fold_metrics["All features baseline"].append(
            metric_values(y_valid, baseline.predict(X_valid))
        )

        # Imputation and selector fitting use only this outer training fold.
        selector_imputer = make_mice_imputer()
        X_train_imp = selector_imputer.fit_transform(X_train.to_numpy(copy=True))
        for method, count in FEATURE_METHODS:
            selector = make_selector(method)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selector.fit(X_train_imp, y_train)
            importances = np.asarray(selector.feature_importances_, dtype=float)
            chosen_idx = np.argsort(importances)[::-1][:count]
            for rank, index in enumerate(chosen_idx, start=1):
                selected_rows.append({
                    "method": method,
                    "fold": fold_number,
                    "rank": rank,
                    "feature": X.columns[index],
                    "importance": importances[index],
                })

            classifier = make_classifier()
            classifier.fit(X_train.iloc[:, chosen_idx], y_train)
            predictions = classifier.predict(X_valid.iloc[:, chosen_idx])
            fold_metrics.setdefault(method, []).append(metric_values(y_valid, predictions))

    summaries = []
    for method, metrics_per_fold in fold_metrics.items():
        row = {
            "method": method,
            "features": X.shape[1] if method == "All features baseline" else dict(FEATURE_METHODS)[method],
        }
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = [fold[metric] for fold in metrics_per_fold]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1))
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "feature_selection_results.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(RESULTS / "selected_features.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    best = summary.loc[summary["mean_macro_f1"].idxmax()]
    baseline = summary.loc[summary["method"] == "All features baseline"].iloc[0]
    print(
        f"\nBest observed by mean macro-F1: {best['method']} "
        f"({best['mean_macro_f1']:.4f}); baseline: {baseline['mean_macro_f1']:.4f}."
    )
    print("This is a preliminary binary LGG-versus-GBM source-file experiment, not the original CliRad three-class study.")
    print(f"Saved: {RESULTS / 'feature_selection_results.csv'}")
    print(f"Saved: {RESULTS / 'selected_features.csv'} (fold-specific ranks and importances)")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Feature comparison failed: {exc}")
