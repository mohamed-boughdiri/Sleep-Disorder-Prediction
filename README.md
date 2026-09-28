# Sleep Disorder Classification

Predicting Sleep Disorder (None / Insomnia / Sleep Apnea) from demographic, lifestyle, and health data, using a three-stage feature selection pipeline (filter methods → Random Forest importance → RFECV) followed by a Random Forest classifier.

## Dataset

[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) (Kaggle) — 374 records, 13 columns covering demographics (Age, Gender, Occupation), lifestyle (Sleep Duration, Physical Activity Level, Daily Steps, Stress Level), and health metrics (BMI Category, Blood Pressure, Heart Rate, Sleep Disorder).

## Pipeline

1. **Data cleaning** — drop identifier column, fill missing `Sleep Disorder` values as `"none"`, merge duplicate BMI labels.
2. **EDA** — target distribution, demographic breakdowns, correlation checks.
3. **Feature engineering & encoding** — split Blood Pressure into Systolic/Diastolic, drop redundant correlated columns, encode categoricals by type (binary / ordinal / one-hot).
4. **Filter-based feature selection** — ANOVA F-test and mutual information (numeric features), chi-square (categorical features).
5. **Train/test split** — stratified, to preserve class balance.
6. **Model-based feature importance** — Random Forest `.feature_importances_`.
7. **Recursive Feature Elimination (RFECV)** — cross-validated, data-driven optimal feature count.
8. **Final model & evaluation** — classification report, confusion matrix, cross-validated F1-macro.

## Results

- **15 final features** selected via RFECV.
- **92% accuracy**, **0.90 macro F1** on the held-out test set.
- **0.86 ± 0.05 macro F1** across 5-fold stratified cross-validation.
- Misclassifications are concentrated between Insomnia and Sleep Apnea (physiologically overlapping conditions), rarely between "no disorder" and an actual disorder.

## Setup

```bash
pip install -r requirements.txt
```

Place `Sleep_health_and_lifestyle_dataset.csv` in the project root, then run `sleep_disorder_analysis.ipynb` top to bottom.

## Notes

The dataset is small (374 rows) with imbalanced classes (~58% None / ~21% each disorder) — the cross-validated score is the more reliable performance estimate, not the single test-split number.
