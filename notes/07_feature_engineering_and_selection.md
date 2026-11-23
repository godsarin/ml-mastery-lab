# Note 07: Feature Engineering and Feature Selection

> Read this note first. Then open `notebooks/01_supervised_churn/07_feature_engineering_and_selection.ipynb`.

---

## Part A: Feature Engineering

### 1. What is it?

**Feature engineering** means creating new, more useful columns from the columns you already have, using common sense about the problem.

Think of a doctor. A patient's weight and height are useful, but the doctor computes **BMI = weight / height²** because that single number tells more than the two separately. BMI is an engineered feature.

### 2. Why do we need it?

- Models only see the columns you give them. They cannot always discover obvious ideas on their own, especially simple models.
- A good feature can improve a simple model more than switching to a fancier algorithm.
- It lets you put your **domain knowledge** (how the business works) into the data.

### 3. How does it work?

Common ways to create features:

| Technique | Example in churn data |
|---|---|
| **Counting** | Number of add-on services a customer has |
| **Combining columns** | `MonthlyCharges × tenure` |
| **Ratios / differences** | Average spend per month = `TotalCharges / tenure` |
| **Binning** (grouping numbers into ranges) | `tenure` → 0-12, 13-24, 25-48, 49+ months |
| **Flags from text** | "Does the payment method contain the word *automatic*?" → 1/0 |
| **Date parts** | Month, weekday (not in this dataset) |

The recipe: **have an idea → build the feature → check that it relates to the target → keep it only if it adds something.** Many ideas will be useless or redundant, and that is normal.

## Part B: Feature Selection

### 1. What is it?

**Feature selection** means choosing which features to keep and which to remove. Less is often better: fewer columns that carry real information.

### 2. Why do we need it?

- **Noise:** useless features add random patterns that the model may wrongly learn (overfitting).
- **Redundancy:** two columns saying the same thing add nothing, and can make linear models unstable.
- **Speed and simplicity:** fewer features train faster and are easier to explain.
- **The curse of dimensionality:** distance-based models such as KNN work worse as the number of columns grows, because every point becomes "far" from every other.

### 3. How does it work?

Three families of methods:

| Family | Idea | Example | When |
|---|---|---|---|
| **Filter** | Score each feature on its own against the target, *before* training any model | Correlation, **mutual information**, chi-square | Fast first look (**we use this now**) |
| **Wrapper** | Train a model repeatedly with different feature sets and keep the best set | Recursive Feature Elimination (RFE) | Small datasets, when you can afford the time |
| **Embedded** | The model itself reveals which features matter while training | Random Forest feature importance, Lasso | Later notebooks (10, 11) |

**Mutual information (MI)** in simple words: "how much does knowing this feature reduce my uncertainty about the target?"
- MI = 0: the feature tells you nothing about the target.
- Higher MI: the feature is more informative.
- Unlike correlation, MI can catch **non-straight-line** relationships, and it works for categories too.

**Redundancy check:** if two numeric features have correlation near ±1 (say above 0.9), they carry the same information. Keep one.

## 4. When do we use it?

- Engineering: after EDA, once you understand the problem and have ideas.
- Selection: after the features exist and are in numeric form, and **always using the training set only**.
- You can repeat both when a model's results suggest a problem.

## 5. Important terminology

| Term | Simple meaning |
|---|---|
| **Feature engineering** | Creating new features from existing ones |
| **Domain knowledge** | Understanding of how the real-world business works |
| **Binning** | Turning a number into groups (ranges) |
| **Feature selection** | Choosing which features to keep |
| **Filter / wrapper / embedded** | Three families of selection methods (see above) |
| **Mutual information** | A score of how informative a feature is about the target (0 = nothing) |
| **Redundant feature** | A feature whose information is already contained in another |
| **Multicollinearity** | Features that are strongly related to each other |
| **Dimensionality** | The number of features (columns) |
| **Curse of dimensionality** | Too many features make distance-based learning worse |
| **Target leakage** | A feature that secretly contains the answer, or information from after the outcome |
| **Rule-based feature** | A feature computed from a fixed rule on one row, learning nothing from the data |

## 6. Simple example

Three customers:

| tenure | OnlineSecurity | TechSupport | Churn |
|---|---|---|---|
| 3 | No | No | Yes |
| 40 | Yes | Yes | No |
| 8 | No | Yes | No |

New feature idea: `num_addons` = number of "Yes" among the add-ons → `0, 2, 1`.
Pattern: more add-ons, less churn. One simple number now summarises several columns.

Selection example: a column `favourite_colour` whose MI with churn is about 0 → drop it.

## 7. Connection to the ML workflow

```
... → Split → Encode/Scale → [7. Engineer + Select] → Pipeline → Train → ...
                                   ^ you are here
```

### The leakage rules for this notebook

1. **Rule-based engineering is safe anywhere.** Counting add-ons uses only that row. It learns nothing from other rows, so the same function can be applied to training data, test data and a brand-new customer.
2. **Data-driven steps must use the training set only.** Mutual information, correlation filters and importance scores are *learned from the target*. Computing them on the full data lets the test labels influence which features you keep. We compute them on `X_train` and `y_train` only.
3. **Target leakage:** never build a feature using the target (for example "churn rate of this customer's contract type"), and never use information that would not exist when predicting.

## 8. Common mistakes

1. **Building features from the target.** The model cheats, scores look amazing, and it fails in real life.
2. **Selecting features on the whole dataset.** Leakage (see above).
3. **Trusting one score blindly.** A feature with MI near 0 on its own can still help in combination with others. Treat selection as evidence, not proof.
4. **Dropping a feature only because it has low correlation.** Correlation only sees straight-line links.
5. **Making hundreds of random features.** More is not better. Every feature needs a reason.
6. **Forgetting to apply the same engineering to the test set and new data.** The model expects the same columns every time. Keep the logic in one function.
7. **Ignoring confounding.** A feature can look strange because it mixes groups (we will meet an example).

## 9. Key takeaways

- **Engineering** creates features from domain ideas: counts, ratios, bins, flags.
- **Selection** removes noise and redundancy. Fewer, better features.
- Use **filter methods** (mutual information, correlation) as a first pass. Wrapper and embedded methods come later.
- **Rule-based features** are safe to build before the split. **Data-driven scores** must come from the training set only.
- Never use the target to build features.
- Keep the engineering code in **one reusable function** so training, testing and production use identical steps.
- Low MI is a hint to drop a feature, not a proof. We confirm with model results later.

---

**Next:** open the notebook for this note and run it cell by cell.
