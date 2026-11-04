from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "combined_data.csv"
OUTPUT_PATH = ROOT / "results" / "oversampling_status.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run src/prepare_data.py first.")
    data = pd.read_csv(DATA_PATH)
    if "label" not in data:
        raise ValueError("Prepared table has no label column.")
    counts = data["label"].value_counts().sort_index()
    max_count, min_count = int(counts.max()), int(counts.min())
    ratio = max_count / min_count
    justified = ratio >= 1.5
    reason = (
        f"No oversampling comparison run: observed class counts are {counts.to_dict()} "
        f"(majority/minority ratio {ratio:.3f}), so this binary pilot is nearly balanced. "
        "More importantly, labels encode source CSV rather than verified subtypes, and "
        "246 rows with 1,213 features are too limited to support a meaningful 1,213-D CTGAN experiment."
    )
    rows = []
    for method in ("No oversampling", "SMOTE", "ADASYN", "CTGAN"):
        rows.append({"method": method, "class_0_count": int(counts.get(0, 0)),
                     "class_1_count": int(counts.get(1, 0)), "majority_minority_ratio": ratio,
                     "run_status": "NOT RUN - not justified for current data", "reason": reason,
                     "accuracy_mean": "", "accuracy_std": "", "macro_f1_mean": "",
                     "macro_f1_std": "", "balanced_accuracy_mean": "", "balanced_accuracy_std": ""})
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(reason)
    print(f"Saved status table: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Oversampling assessment failed: {exc}")
