from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from compare_features import make_selector
from model_utils import make_mice_imputer, writable_array

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_PATH = RESULTS / "combined_data.csv"
SEED = 42

MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees",
    "SVM", "KNN", "QDA", "AdaBoost", "Gradient Boosting", "XGBoost",
    "XGBoost RF", "CatBoost", "LightGBM",
]


def suggest_parameters(name, trial, min_train_rows):
    if name == "Logistic Regression":
        return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
    if name == "Decision Tree":
        return {"max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8)}
    if name in {"Random Forest", "Extra Trees"}:
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)}
    if name == "SVM":
        return {"C": trial.suggest_float("C", 0.1, 10.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])}
    if name == "KNN":
        upper = max(2, min(25, min_train_rows - 1))
        return {"n_neighbors": trial.suggest_int("n_neighbors", 2, upper),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
    if name == "QDA":
        return {"reg_param": trial.suggest_float("reg_param", 0.0, 1.0)}
    if name == "AdaBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 30, 120, step=30),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 1.0, log=True)}
    if name == "Gradient Boosting":
        return {"n_estimators": trial.suggest_int("n_estimators", 30, 120, step=30),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 1, 3)}
    if name in {"XGBoost", "XGBoost RF"}:
        params = {"n_estimators": trial.suggest_int("n_estimators", 30, 120, step=30),
                  "max_depth": trial.suggest_int("max_depth", 2, 5),
                  "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True)}
        if name == "XGBoost RF":
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["colsample_bynode"] = trial.suggest_float("colsample_bynode", 0.6, 1.0)
        return params
    if name == "CatBoost":
        return {"iterations": trial.suggest_int("iterations", 50, 150, step=50),
                "depth": trial.suggest_int("depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True)}
    if name == "LightGBM":
        return {"n_estimators": trial.suggest_int("n_estimators", 30, 120, step=30),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True)}
    raise ValueError(f"No search space is defined for {name}.")


def make_estimator(name, params):
    common_tree = {"random_state": SEED}
    if name == "Logistic Regression":
        return LogisticRegression(C=params["C"], max_iter=2000, solver="liblinear", class_weight="balanced", **common_tree)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"], class_weight="balanced", **common_tree)
    if name == "Random Forest":
        return RandomForestClassifier(**params, class_weight="balanced", n_jobs=1, **common_tree)
    if name == "Extra Trees":
        return ExtraTreesClassifier(**params, class_weight="balanced", n_jobs=1, **common_tree)
    if name == "SVM":
        return SVC(**params, class_weight="balanced", random_state=SEED)
    if name == "KNN":
        return KNeighborsClassifier(**params, n_jobs=1)
    if name == "QDA":
        return QuadraticDiscriminantAnalysis(**params)
    if name == "AdaBoost":
        return AdaBoostClassifier(**params, random_state=SEED)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(**params, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED)
    if name == "XGBoost RF":
        forest_params = dict(params)
        parallel_trees = forest_params.pop("n_estimators")
        return XGBClassifier(**forest_params, n_estimators=1, num_parallel_tree=parallel_trees, objective="binary:logistic", eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED)
    if name == "CatBoost":
        return CatBoostClassifier(**params, verbose=False, thread_count=1, allow_writing_files=False, random_seed=SEED)
    if name == "LightGBM":
        return LGBMClassifier(**params, n_jobs=1, verbosity=-1, random_state=SEED)
    raise ValueError(f"Unknown classifier: {name}")


def make_pipeline_for(name, params):
    return make_pipeline(FunctionTransformer(writable_array, validate=False), make_mice_imputer(), StandardScaler(), make_estimator(name, params))


def scores(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}


def select_xgb_dt_features(X_train, y_train, feature_count=21):
    imputer = make_mice_imputer()
    values = imputer.fit_transform(X_train.to_numpy(copy=True))
    selector = make_selector("XGB-DT")
    selector.fit(values, y_train)
    chosen = np.argsort(selector.feature_importances_)[::-1][:feature_count]
    return X_train.columns[chosen].tolist()


def tune_in_inner_cv(name, X, y, trials, inner_folds, outer_fold):
    min_count = int(y.value_counts().min())
    n_splits = min(inner_folds, min_count)
    if n_splits < 2:
        raise ValueError("Not enough examples per class for inner stratified CV.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + outer_fold)
    # Rank features separately inside each inner training fold. Cache these fold
    # matrices because the selector does not depend on the classifier trial.
    inner_fold_data = []
    for train_idx, valid_idx in cv.split(X, y):
        X_inner_train, X_inner_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_inner_train, y_inner_valid = y.iloc[train_idx], y.iloc[valid_idx]
        selected = select_xgb_dt_features(X_inner_train, y_inner_train)
        inner_fold_data.append((
            X_inner_train[selected], y_inner_train,
            X_inner_valid[selected], y_inner_valid,
        ))

    def objective(trial):
        min_train_rows = min(len(fold[1]) for fold in inner_fold_data)
        params = suggest_parameters(name, trial, min_train_rows)
        fold_scores = []
        for X_inner_train, y_inner_train, X_inner_valid, y_inner_valid in inner_fold_data:
            model = make_pipeline_for(name, params)
            model.fit(X_inner_train, y_inner_train)
            prediction = model.predict(X_inner_valid)
            fold_scores.append(scores(y_inner_valid, prediction)["macro_f1"])
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + outer_fold))
    study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def run(args):
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run src/prepare_data.py first.")
    data = pd.read_csv(DATA_PATH)
    if "label" not in data:
        raise ValueError("Prepared table has no 'label' column.")
    y = data.pop("label").astype(int)
    metadata_names = {"patient_id", "id", "subject_id", "case_id", "sample_id"}
    feature_data = data.drop(columns=[c for c in data.columns if str(c).strip().lower() in metadata_names])
    X = feature_data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    counts = y.value_counts()
    if y.nunique() != 2 or counts.min() < args.outer_folds:
        raise ValueError("Pilot requires two classes and at least one sample per fold in both classes.")
    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=SEED)
    names = [name.strip() for name in args.models.split(",")] if args.models else MODEL_NAMES
    unknown = sorted(set(names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")

    fold_rows, summary_rows, trial_rows = [], [], []
    for name in names:
        print(f"\nModel: {name}", flush=True)
        model_fold_scores = []
        model_fold_params = []
        for outer_fold, (train_idx, valid_idx) in enumerate(outer.split(X, y), start=1):
            print(f"  outer fold {outer_fold}/{args.outer_folds}: tune on training rows", flush=True)
            X_train_all, X_valid_all = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            best_params, inner_macro_f1 = tune_in_inner_cv(
                name, X_train_all, y_train, args.trials, args.inner_folds, outer_fold
            )
            # Refit selection on the complete outer training fold only.
            selected = select_xgb_dt_features(X_train_all, y_train)
            X_train, X_valid = X_train_all[selected], X_valid_all[selected]
            final_fold_model = make_pipeline_for(name, best_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_fold_model.fit(X_train, y_train)
            result = scores(y_valid, final_fold_model.predict(X_valid))
            model_fold_scores.append(result)
            model_fold_params.append(best_params)
            fold_rows.append({"model": name, "outer_fold": outer_fold, "selected_feature_count": len(selected),
                              "inner_best_macro_f1": inner_macro_f1, **result,
                              "best_params": json.dumps(best_params, sort_keys=True)})
            trial_rows.append({"model": name, "outer_fold": outer_fold, "trials_requested": args.trials,
                               "best_inner_macro_f1": inner_macro_f1,
                               "best_params": json.dumps(best_params, sort_keys=True)})
        row = {"model": name, "feature_set": "XGB-DT top 21 (fit in each outer training fold)",
               "selected_features": 21, "outer_folds": args.outer_folds, "trials_per_outer_fold": args.trials}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = [entry[metric] for entry in model_fold_scores]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["mean_best_inner_macro_f1"] = float(np.mean([r["best_inner_macro_f1"] for r in trial_rows if r["model"] == name]))
        row["fold_best_params_json"] = json.dumps(model_fold_params, sort_keys=True)
        summary_rows.append(row)

    old_feature_results = RESULTS / "feature_selection_results.csv"
    if old_feature_results.exists():
        previous = pd.read_csv(old_feature_results)
        for method, model_name in (("All features baseline", "All-feature Logistic Regression baseline"),
                                   ("XGB-DT", "Logistic Regression with XGB-DT top 21")):
            match = previous.loc[previous["method"] == method]
            if not match.empty:
                source = match.iloc[0]
                summary_rows.append({
                    "model": model_name,
                    "feature_set": "existing five-fold feature-selection comparison",
                    "selected_features": int(source["features"]),
                    "outer_folds": 5,
                    "trials_per_outer_fold": 0,
                    "mean_accuracy": source["mean_accuracy"],
                    "std_accuracy": source["std_accuracy"],
                    "mean_macro_f1": source["mean_macro_f1"],
                    "std_macro_f1": source["std_macro_f1"],
                    "mean_balanced_accuracy": source["mean_balanced_accuracy"],
                    "std_balanced_accuracy": source["std_balanced_accuracy"],
                    "mean_best_inner_macro_f1": "",
                    "fold_best_params_json": "not tuned; prior five-fold feature comparison",
                })
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "model_comparison_folds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(RESULTS / "model_comparison_results.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(RESULTS / "optuna_best_parameters.csv", index=False)
    print("\nNested-CV model summary (experimental binary pilot):")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved model comparison, fold metrics and Optuna parameters to {RESULTS}")


def parse_args():
    parser = argparse.ArgumentParser(description="Nested-CV Optuna classifier comparison for the provisional binary pilot.")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per classifier per outer fold (paper reports 20).")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--models", default="", help="Optional comma-separated subset used for a smoke test.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Model experiment failed: {exc}")





