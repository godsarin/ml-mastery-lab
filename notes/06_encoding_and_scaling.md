# Note 06: Categorical Encoding and Feature Scaling

> Read this note first. Then open `notebooks/01_supervised_churn/06_encoding_and_scaling.ipynb`.

---

## Part A: Categorical Encoding

### 1. What is it?

Machine learning models do maths, so they only understand **numbers**. Our data contains text such as `"Month-to-month"`, `"Fiber optic"` or `"Yes"`. **Encoding** means converting those text categories into numbers in a way that does not mislead the model.

### 2. Why do we need it?

Scikit-learn models crash if you give them text. And a careless conversion can mislead the model. If you simply number `Fiber optic = 2, DSL = 1, No = 0`, the model may think fiber is "twice" DSL, or that the categories have an order that does not exist.

### 3. How does it work?

| Method | What it does | Example | Use when |
|---|---|---|---|
| **Binary (0/1) encoding** | Two categories become 0 and 1 | `No → 0`, `Yes → 1` | Exactly two categories |
| **One-hot encoding** | One new 0/1 column per category | `Contract` → `Contract_Month-to-month`, `Contract_One year`, `Contract_Two year` | Categories with **no natural order** (*nominal*) |
| **Ordinal encoding** | Categories become 0, 1, 2... in a meaningful order | `Month-to-month=0, One year=1, Two year=2` | Categories with a **real order** (*ordinal*), for example Low < Medium < High |

**One-hot example**

| PaymentMethod | → | is_Electronic check | is_Mailed check | is_Bank transfer |
|---|---|---|---|---|
| Electronic check | | 1 | 0 | 0 |
| Mailed check | | 0 | 1 | 0 |
| Bank transfer | | 0 | 0 | 1 |

Each row has exactly one "hot" (1) column. No category is bigger than another.

**The dummy variable trap:** with 3 categories, 2 columns are enough (if both are 0, it must be the third). Keeping all 3 makes the columns perfectly predictable from each other, which can confuse linear models. Dropping one is the usual fix (`drop="first"`). For simplicity we keep all columns for multi-category features, and drop one only for Yes/No columns (`drop="if_binary"`).

**New categories at prediction time:** tomorrow a customer may arrive with a payment method the model never saw. The encoder must not crash. We use `handle_unknown="ignore"`, which gives that row all zeros for the feature.

## Part B: Feature Scaling

### 1. What is it?

Scaling puts numeric features on a **similar range**, so that no feature dominates just because its numbers are bigger.

`tenure` runs 0 to 72. `TotalCharges` runs 0 to 8,700. Without scaling, a model measuring distance sees `TotalCharges` as 100 times more important, only because of its units.

### 2. Why do we need it?

Some algorithms are sensitive to size:
- **Distance-based:** KNN, SVM. Distance is dominated by the biggest-range feature.
- **Gradient/regularised models:** Logistic Regression and Linear Regression train faster and more stably, and regularisation treats all features fairly only if they share a scale.

Some algorithms **do not care**:
- **Decision Trees and Random Forest**: they split on thresholds, so the units do not matter.
- **Naive Bayes** is also largely unaffected by scale.

Scaling is harmless for trees, so we simply apply it in a pipeline for all models, and we are careful to do it correctly.

### 3. How does it work?

| Method | Formula | Result | Use when |
|---|---|---|---|
| **Standardization** (`StandardScaler`) | (x − mean) / std | Mean 0, std 1. Values are not limited to a range | **Default choice**. Good for Logistic Regression, SVM, KNN |
| **Normalization / Min-Max** (`MinMaxScaler`) | (x − min) / (max − min) | Values between 0 and 1 | You need a fixed range, or data is not bell-shaped |

Example: tenures `[0, 36, 72]`
- Min-Max → `[0, 0.5, 1]`.
- Standardization → about `[-1.22, 0, 1.22]`.

### Important: do not scale…
- **Encoded 0/1 columns.** They are already on a 0-1 scale. Scaling them makes them harder to interpret and adds nothing.
- **The target** `y`, in classification.

## 4. When do we use it?

- Encoding: **always**, whenever you have text categories and a model that needs numbers (all scikit-learn models except a few such as `HistGradientBoosting`).
- Scaling: whenever you use KNN, SVM, Logistic Regression, or neural networks. Optional for trees.
- **Both after the train/test split**, fitted on the training set only (Note 05).

## 5. Important terminology

| Term | Simple meaning |
|---|---|
| **Encoding** | Converting categories to numbers |
| **Nominal** | Categories with no order (colours, payment methods) |
| **Ordinal** | Categories with a natural order (small < medium < large) |
| **One-hot encoding** | One 0/1 column per category |
| **Dummy variable trap** | Redundant columns from keeping every category, which can upset linear models |
| **Sparse / dense** | Sparse = stores only non-zero values to save memory. Dense = a normal table |
| **Scaling** | Changing the range of numeric values |
| **Standardization (z-score)** | Subtract the mean, divide by the standard deviation |
| **Normalization (Min-Max)** | Squeeze into the range 0-1 |
| **Feature magnitude** | How large the numbers in a column are |
| **`ColumnTransformer`** | A scikit-learn tool that applies different steps to different groups of columns |
| **`handle_unknown`** | What the encoder does with a category it never saw |

## 6. Simple example

Two customers, described by `tenure` (months) and `MonthlyCharges` (dollars):

- A: `(tenure 10, charges 50)` B: `(tenure 60, charges 52)`

Distance (straight line): `sqrt(50² + 2²)` = about **50.04**. The huge tenure gap decides everything and the $2 difference hardly counts.
After scaling, both features contribute in proportion to *how unusual* the difference is, not what units they use.

## 7. Connection to the ML workflow

```
... → Split → [6. Encode + Scale (fit on train)] → Feature engineering → Pipeline → Train → ...
                       ^ you are here
```

The three golden steps:
1. `fit` the encoder and scaler on **X_train only**.
2. `transform` X_train.
3. `transform` X_test with the **same fitted objects**.

In this notebook we do them by hand to see what happens. In notebook 08, a **Pipeline** will do all three for us automatically and safely, even inside cross-validation.

## 8. Common mistakes

1. **Fitting on the full data** (or on the test set). This is leakage (Note 05).
2. **Label-encoding nominal categories** (`Fiber=2, DSL=1, No=0`). The model invents an order.
3. **Using one-hot encoding on features with hundreds of categories** (zip codes). It creates huge tables. Other techniques exist, but are not needed here.
4. **Scaling the 0/1 columns** that came from one-hot encoding.
5. **Scaling the target** in classification.
6. **Forgetting that the final model needs the same columns, in the same order.** The saved model must be saved together with its encoder and scaler (a pipeline solves this).
7. **Believing "all models need scaling".** Trees do not.
8. **Not handling unseen categories** in production.

## 9. Key takeaways

- Models need numbers. **Encode** text categories without inventing a false order.
- 2 categories → 0/1. Nominal → **one-hot**. Real order → **ordinal**.
- **Scale** numeric features so that units do not decide importance. Needed by KNN, SVM, Logistic Regression. Not needed by trees.
- **StandardScaler** is the default. MinMaxScaler gives a 0-1 range.
- Fit encoders and scalers on **training data only**. Transform train and test with the same fitted objects.
- Use `handle_unknown="ignore"` so new categories do not crash predictions.
- A `ColumnTransformer` applies the right step to the right columns. Notebook 08 wraps it all in a pipeline.

---

**Next:** open the notebook for this note and run it cell by cell.
