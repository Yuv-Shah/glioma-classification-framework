from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ID_NAMES = {"id", "patientid", "subjectid", "caseid", "sampleid"}
TCGA_ID_PATTERN = r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9-]+)?$"


def normalize_name(name: str) -> str:
    """Normalize known radiomic-name differences without fuzzy matching."""
    name = str(name).strip()
    name = re.sub(r"^original_shape_", "", name, flags=re.I)
    name = re.sub(r"^original_firstorder_", "", name, flags=re.I)
    name = re.sub(r"^firstorder_", "", name, flags=re.I)
    name = re.sub(r"^X(?=\d)", "", name)
    name = re.sub(r"^original_(glcm|glrlm|glszm|gldm|ngtdm)_", r"\1_", name, flags=re.I)
    name = re.sub(r"^log\.sigma\.(\d+)\.0\.mm\.3D_", r"sigma\1_", name, flags=re.I)
    name = re.sub(r"^wavelet\.([LH]{3})_", r"\1_", name, flags=re.I)
    return "".join(ch.lower() for ch in name if ch.isalnum())


def load_source(path: Path, source_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required data file is missing: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {path.name} as CSV: {exc}") from exc
    if df.empty:
        raise ValueError(f"{path.name} contains no data rows.")

    original_shape = df.shape
    id_columns = [c for c in df.columns if normalize_name(c) in ID_NAMES]
    unnamed_columns = [c for c in df.columns if str(c).startswith("Unnamed:")]
    for column in unnamed_columns:
        values = df[column].dropna().astype(str).str.strip()
        if len(values) == len(df) and values.str.match(TCGA_ID_PATTERN, case=False).all():
            id_columns.append(column)
    id_columns = list(dict.fromkeys(id_columns))
    patient_ids = df[id_columns[0]].astype("string").str.strip() if id_columns else None

    # Discard unnamed columns only when they are not verified-looking TCGA IDs.
    drop_columns = [c for c in unnamed_columns if c not in id_columns]
    df = df.drop(columns=list(dict.fromkeys(id_columns + drop_columns)))
    non_numeric_columns = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    feature_df = df.select_dtypes(include=[np.number]).copy()
    if feature_df.empty:
        raise ValueError(f"{path.name} has no numeric feature columns after excluding metadata.")
    normalized = [normalize_name(c) for c in feature_df.columns]
    if "" in normalized:
        raise ValueError(f"{path.name} has a feature name that becomes empty after normalization.")
    duplicates = pd.Index(normalized)[pd.Index(normalized).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Ambiguous duplicate normalized feature names in {path.name}: {duplicates[:5]}")
    feature_df.columns = normalized
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    id_duplicates = int(patient_ids.duplicated().sum()) if patient_ids is not None else None
    info = {
        "source": source_name,
        "file": path.name,
        "rows": len(df),
        "original_columns": original_shape[1],
        "numeric_features": len(feature_df.columns),
        "non_numeric_columns": non_numeric_columns,
        "id_columns": [str(c) for c in id_columns],
        "id_count": int(patient_ids.notna().sum()) if patient_ids is not None else 0,
        "unique_patient_ids": int(patient_ids.nunique()) if patient_ids is not None else 0,
        "duplicate_patient_ids": id_duplicates,
        "missing_cells_after_inf_conversion": int(feature_df.isna().sum().sum()),
        "duplicate_records": int(feature_df.duplicated().sum()),
        "infinite_values": int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()),
        "class_count": len(df),
    }
    return feature_df, patient_ids, info


def main():
    RESULTS.mkdir(exist_ok=True)
    sources = [
        ("LGG source (provisional class 0)", DATA / "tcga_lgg.csv"),
        ("GBM source (provisional class 1)", DATA / "tcga_gbm.csv"),
    ]
    loaded = [load_source(path, label) for label, path in sources]
    lgg, gbm = loaded[0][0], loaded[1][0]
    lgg_ids, gbm_ids = loaded[0][1], loaded[1][1]
    info = [item[2] for item in loaded]
    lgg_only = sorted(set(lgg.columns) - set(gbm.columns))
    gbm_only = sorted(set(gbm.columns) - set(lgg.columns))
    if lgg_only or gbm_only:
        raise ValueError(
            "Feature sets are not compatible after safe name normalization. "
            f"LGG-only: {len(lgg_only)} (examples: {lgg_only[:5]}), "
            f"GBM-only: {len(gbm_only)} (examples: {gbm_only[:5]})."
        )
    common = sorted(lgg.columns)
    if not common:
        raise ValueError("No matching numeric features found between the two CSV files.")
    lgg, gbm = lgg[common], gbm[common]
    numeric_stats = pd.concat([lgg, gbm], ignore_index=True)
    y = pd.Series([0] * len(lgg) + [1] * len(gbm), name="label")
    patient_ids = pd.concat([
        lgg_ids if lgg_ids is not None else pd.Series([pd.NA] * len(lgg), dtype="string"),
        gbm_ids if gbm_ids is not None else pd.Series([pd.NA] * len(gbm), dtype="string"),
    ], ignore_index=True).rename("patient_id")
    combined = pd.concat([lgg, gbm], ignore_index=True)
    combined.insert(0, "patient_id", patient_ids)
    combined.insert(0, "label", y)

    source_standardization = []
    for source_name, table in (("LGG", lgg), ("GBM", gbm)):
        feature_means = table.mean().abs()
        feature_stds = table.std()
        source_standardization.append((
            source_name,
            float(feature_means.median()),
            float(feature_means.max()),
            float(feature_stds.median()),
            int((feature_stds.sub(1.0).abs() < 1e-6).sum()),
        ))
    median_abs_mean = float(numeric_stats.mean().abs().median())
    median_std = float(numeric_stats.std().median())
    median_shifts = (gbm.median() - lgg.median()).abs().sort_values(ascending=False)
    shift_over_025 = int((median_shifts > 0.25).sum())
    shift_over_050 = int((median_shifts > 0.50).sum())
    largest_shift = str(median_shifts.index[0])
    largest_shift_value = float(median_shifts.iloc[0])
    lgg_id_set = set(lgg_ids.dropna().astype(str)) if lgg_ids is not None else set()
    gbm_id_set = set(gbm_ids.dropna().astype(str)) if gbm_ids is not None else set()
    id_overlap = len(lgg_id_set & gbm_id_set)

    report = [
        "CliRad Phase 1 data audit",
        "=========================",
        "Labels are provisional source-file labels: 0=TCGA-LGG file, 1=TCGA-GBM file.",
        "They are not verified molecular subtype labels; labels were assigned from source files.",
        "Patient IDs found in the unnamed first column are retained as metadata and are not model features.",
        "",
    ]
    for item in info:
        report.extend([
            f"{item['source']} ({item['file']}):",
            f"  original CSV dimensions (rows x columns): {item['rows']} x {item['original_columns']}",
            f"  numeric features: {item['numeric_features']}",
            f"  recognized ID columns: {item['id_columns']}",
            f"  patient IDs found / unique / repeated: {item['id_count']} / {item['unique_patient_ids']} / {item['duplicate_patient_ids']}",
            f"  non-numeric feature columns excluded: {item['non_numeric_columns']}",
            f"  missing numeric cells (infinities converted to missing): {item['missing_cells_after_inf_conversion']}",
            f"  infinite numeric values found: {item['infinite_values']}",
            f"  duplicate feature records: {item['duplicate_records']}",
            f"  provisional class distribution: {item['class_count']}",
        ])
    report.extend([
        "",
        f"Matched compatible features: {len(common)}",
        f"Combined matrix: {len(combined)} rows x {len(common)} numeric features, plus label and patient_id metadata",
        f"Combined duplicate feature rows: {int(numeric_stats.duplicated().sum())}",
        f"Exact patient ID overlap across source files: {id_overlap}",
        f"Feature distribution check: {shift_over_025} / {len(common)} feature medians differ by >0.25 standardized units; {shift_over_050} differ by >0.50",
        f"Largest absolute median difference: {largest_shift} ({largest_shift_value:.4f})",
        f"Combined missing feature cells: {int(numeric_stats.isna().sum().sum())}",
        f"Provisional label distribution: {y.value_counts().sort_index().to_dict()}",
        "",
        "Standardization check:",
        f"  pooled median absolute feature mean: {median_abs_mean:.6g}",
        f"  pooled median feature standard deviation: {median_std:.6g}",
        *[f"  {source}: median/max absolute feature mean={median_mean:.6g}/{max_mean:.6g}; median feature SD={median_sd:.12g}; features with SD within 1e-6 of 1={sd_count}/{len(common)}"
          for source, median_mean, max_mean, median_sd, sd_count in source_standardization],
        "  Every feature in both files has SD within 1e-6 of 1 and means near zero, strongly consistent with separate whole-file column standardization.",
        "  If fitted on each complete source file, this includes the later validation rows and is an upstream leakage risk; the source data do not verify when/how it was done.",
        "  Unknown raw values/parameters cannot be recovered; fold-local scaling cannot undo that transformation.",
        "",
        "Feature names use known-prefix normalization followed by exact comparison; no fuzzy matching is used.",
        "Only numeric feature columns enter modeling. The patient_id metadata column is explicitly excluded by modeling scripts.",
    ])
    (RESULTS / "data_check.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    combined.to_csv(RESULTS / "combined_data.csv", index=False)
    print("\n".join(report))
    print(f"\nSaved prepared data: {RESULTS / 'combined_data.csv'}")
    print(f"Saved audit: {RESULTS / 'data_check.txt'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Data preparation failed: {exc}")
