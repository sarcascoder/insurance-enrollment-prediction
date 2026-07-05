# Report — Predicting Insurance Enrollment

## 1. Objective

Predict whether an employee opts in (`enrolled = 1`) to a voluntary insurance
product from demographic and employment attributes. This is a **binary
classification** problem, evaluated end-to-end: data processing → modelling →
evaluation → serving.

---

## 2. Data observations

The dataset (`data/employee_data.csv`) has **10,000 rows × 10 columns**.
Reproduce these with `python -m src.eda`.

### Quality
- **No missing values**, **no duplicate `employee_id`s**, and **no conflicting
  duplicate feature rows**. It is a clean synthetic dataset — no imputation was
  needed.
- `employee_id` is a pure identifier and is **dropped** before modelling; keeping
  it would invite memorisation of individual rows.

### Target balance
- `enrolled = 1`: **61.7%**, `enrolled = 0`: **38.3%**. Mildly imbalanced —
  enough to prefer ROC-AUC / F1 over raw accuracy and to use
  `class_weight="balanced"` in the linear/forest models.

### Numeric features
| Feature | Mean | Std | Min | Median | Max |
|---------|-----:|----:|----:|-------:|----:|
| `age` | 43.0 | 12.3 | 22 | 43 | 64 |
| `salary` | 65,033 | 14,924 | 2,208 | 65,056 | 120,312 |
| `tenure_years` | 3.97 | 3.90 | 0 | 2.8 | 36 |

### Which features actually carry signal
Enrollment rate broken down by category is the most revealing view:

| Feature | Strongest → weakest group (enrollment rate) | Signal |
|---------|----------------------------------------------|--------|
| **`has_dependents`** | Yes **0.80** vs No **0.35** | **Strong** |
| **`employment_type`** | Full-time **0.75** vs Contract 0.31 vs Part-time 0.28 | **Strong** |
| `salary` | higher salary → higher enrollment (continuous) | Moderate |
| `age` | mild effect | Weak |
| `gender` | Other 0.64 / Male 0.62 / Female 0.62 | ~None |
| `marital_status` | 0.60–0.63 across all | ~None |
| `region` | 0.61–0.63 across all | ~None |

**Takeaway:** enrollment is driven mostly by `has_dependents`, `employment_type`,
and `salary`. `gender`, `marital_status`, and `region` are essentially noise —
their per-group rates barely move from the 61.7% base rate.

Figures generated for reference:
`artifacts/enrollment_by_category.png`, `artifacts/numeric_distributions.png`.

---

## 3. Data processing pipeline

Implemented in `src/data.py` and `src/preprocess.py`:

1. **Load & validate** — assert the schema, a non-null binary target, and fail
   loudly otherwise (so a bad file never silently corrupts training).
2. **Stratified 80/20 split** — preserves the 61.7% positive rate in both train
   and test, keeping the held-out evaluation representative.
3. **Preprocessing as a `ColumnTransformer`**, wrapped *inside* each model
   `Pipeline`:
   - numeric (`age`, `salary`, `tenure_years`) → `StandardScaler`
   - categorical (`gender`, `marital_status`, `employment_type`, `region`,
     `has_dependents`) → `OneHotEncoder(handle_unknown="ignore")`

**Why preprocessing lives inside the pipeline:** the scaler and encoder are fit
only on the training folds during cross-validation — never on the test set — so
there is **no data leakage**. The same fitted transform is serialised with the
model, so the API receives raw JSON and applies the identical transformation at
inference time.

---

## 4. Model choices & rationale

Three models were trained, each tuned with **5-fold cross-validated grid search**
optimising ROC-AUC (`src/train.py`):

| Model | Why it was chosen |
|-------|-------------------|
| **Logistic Regression** | Fast, interpretable linear baseline. If a linear model does well, the signal is largely additive. |
| **Random Forest** | Captures non-linearities and feature interactions with little tuning; robust to mixed feature types. |
| **HistGradientBoosting** | scikit-learn's boosted trees — usually the strongest tabular model, and built in (no extra dependency such as XGBoost). |

Selection rule: **best test ROC-AUC wins** and is saved to `models/model.joblib`.

---

## 5. Evaluation results

Held-out test set (2,000 rows), from `artifacts/metrics.json`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| Logistic Regression | 0.894 | 0.936 | 0.889 | 0.912 | 0.971 |
| **Random Forest (selected)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| HistGradientBoosting | 0.9995 | 1.000 | 0.999 | 1.000 | 1.000 |

Diagnostic figures for the selected model:
`artifacts/confusion_matrix.png`, `artifacts/roc_curve.png`.

### Why the tree models score ~perfectly (and why that's not leakage)
This is the most important finding. I investigated the perfect scores rather than
accepting them:

- A plain **depth-6 decision tree perfectly classifies all 10,000 rows** (100%
  train accuracy), and a depth-3 tree already reaches 91%.
- There are **zero feature-identical rows with conflicting labels**, i.e. the
  target is a (near-)**deterministic function** of the features.
- The splits the tree finds are thresholds on `salary`, and flags on
  `employment_type = Full-time` and `has_dependents`, plus a little `age` — the
  same features the EDA flagged as strong.

So the synthetic data was generated by an essentially rule-based process, and
tree ensembles recover that rule exactly. My pipeline provably has **no
leakage** (`employee_id` and the target are dropped; preprocessing is fit only
on training folds). The perfect score is a property of the **synthetic
generator**, not a modelling artefact.

**Logistic Regression "only" reaches 0.97 AUC** precisely because the decision
boundary is a set of axis-aligned thresholds and interactions that a single
linear boundary cannot represent — a textbook illustration of linear vs.
tree-based model capacity.

---

## 6. Key takeaways

- Enrollment is predictable primarily from `has_dependents`, `employment_type`,
  and `salary`; three of the eight features are noise.
- On this synthetic data, tree ensembles are effectively perfect and Logistic
  Regression is a strong, interpretable ~0.97-AUC baseline.
- **Perfect metrics are a signal to investigate, not to celebrate.** I verified
  the cause (a deterministic synthetic target) instead of assuming leakage or
  declaring success. On real enrollment data I would expect materially lower,
  noisier numbers.

---

## 7. What I'd do next with more time

1. **Interpretability** — SHAP values / permutation importance on the chosen
   model to quantify each feature's contribution for stakeholders.
2. **Calibration** — for a "likelihood of enrollment" product, calibrated
   probabilities matter more than hard labels; check a reliability curve and
   apply isotonic/Platt scaling if needed.
3. **Threshold selection** — tune the decision threshold to the business cost of
   false positives vs. false negatives rather than defaulting to 0.5.
4. **Robustness to messier data** — the real world has missing values, typos in
   categoricals, and outliers; add imputation and stress-test the pipeline.
5. **Drop the noise features** and confirm no performance loss — a simpler model
   is cheaper to serve and easier to explain.
6. **Stronger validation on real data** — nested CV and a temporal/holdout split
   to get an honest generalisation estimate once scores aren't saturated.
7. **Productionisation** — containerise the API, add request logging and model
   versioning, and wire the MLflow model registry into deployment.

---

## Appendix — reproduce everything

```bash
pip install -r requirements.txt
python -m src.eda        # EDA stats + figures
python -m src.train      # train, tune, track (MLflow), select & save best
python -m pytest -q      # tests
uvicorn src.api:app      # serve predictions at /docs
```
