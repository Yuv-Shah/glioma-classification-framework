# CliRad Phase 1 pilot

This project starts from existing TCGA-LGG and TCGA-GBM radiomic feature CSVs. It audits and aligns them, compares four fold-local feature selectors with Logistic Regression, and runs a separate nested-CV Optuna comparison of 13 student-defined classifiers.

**Scientific scope:** the labels are provisional file-source labels (LGG=0, GBM=1), not verified molecular subtype labels. The project has no MRI volumes or tumor masks, so it does not extract radiomics from images. Current class counts (126 and 120) are nearly balanced; `compare_oversampling.py` records why oversampling was not run. The pilot is not an exact reproduction or clinical model.

## Windows PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands from the project root:

```powershell
python src/prepare_data.py
python src/compare_features.py
python src/train_models.py --trials 20 --outer-folds 5 --inner-folds 3
python src/compare_oversampling.py
```

The final command audits whether oversampling is warranted; it does not create synthetic observations for the current dataset. No prediction command is provided because the source labels and model results do not support meaningful patient-level inference.

## Scripts and outputs

- `src/prepare_data.py`: audits and aligns the input tables; writes `results/data_check.txt` and `results/combined_data.csv`.
- `src/compare_features.py`: compares all-feature Logistic Regression with XGB-DT (21), XGB-RF (37), CatBoost (71), and LightGBM (82) features. Writes `results/feature_selection_results.csv` and `results/selected_features.csv`.
- `src/train_models.py`: performs nested five-fold evaluation with three-fold Optuna tuning. The default requests 20 trials per model per outer fold. Writes `results/model_comparison_results.csv`, `results/model_comparison_folds.csv` and `results/optuna_best_parameters.csv`.
- `src/compare_oversampling.py`: writes `results/oversampling_status.csv` with the class counts and reason samplers were not run.
- `src/model_utils.py`: provides training-fold MICE-style iterative imputation. Complete input data are unchanged.

Patient IDs are retained as metadata in the prepared CSV and excluded from model features. The nested experiment selects XGB-DT features separately inside each inner training fold. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for actual results and the remaining upstream standardization risk. The paper’s full 13-classifier list is not enumerated in the accessible main text; the implemented list is explicitly a student-defined alternative.

