# Note 05: Train/Test Split and Data Leakage

> Read this note first. Then open `notebooks/01_supervised_churn/05_train_test_split.ipynb`.

---

## 1. What is it?

Before training a model, we split our labelled data into two separate parts:

- **Training set**: the data the model **learns from**.
- **Test set**: data the model **never sees** while learning. We use it at the very end to check how well the model works on new customers.

Think of a student preparing for an exam. The textbook and practice questions are the training set. The real exam, with questions the student has never seen, is the test set. If the student had seen the exam questions in advance, a high score would prove nothing.

## 2. Why do we need it?

A model can **memorise** its training data and still be useless on new data. If we test it on the same data it learned from, it will look brilliant. This is the most common way beginners fool themselves.

The test set gives an **honest estimate** of how the model will behave in the real world, on customers it has never met.

## 3. How does it work?

1. Separate the **features (X)** from the **target (y)**.
2. Randomly shuffle the rows and cut them into two parts, usually **80% train / 20% test** (or 75/25, or 70/30).
3. Train the model on the training part only.
4. Test it once on the test part.

In scikit-learn this is one function: `train_test_split`. Three of its settings matter:

| Setting | What it does | Why |
|---|---|---|
| `test_size=0.2` | 20% of rows go to the test set | Enough rows to test, enough left to learn |
| `random_state=42` | Fixes the shuffle | **Reproducibility**: same split every run, so results can be compared. (42 is just a popular number. Any fixed number works) |
| `stratify=y` | Keeps the same class proportions in both parts | Important when classes are imbalanced (here 73% / 27%) |

### Data leakage: the most important idea here

**Data leakage** means information from outside the training set sneaks into the training process. The model then looks better in testing than it will in real life.

Two common forms:

1. **Target leakage**: a feature contains the answer or is created after the outcome. Example: a column `cancellation_date` when predicting churn.
2. **Train/test contamination**: you calculate something from the *whole* dataset (a mean, a median, a min/max, scaling factors, outlier fences) and then split. The test rows have influenced the numbers used in training.

**The golden rule: split first. Then learn everything (every mean, median, scale, encoding, selection) from the training set only, and apply it unchanged to the test set.**

In scikit-learn terms:
- `fit` = learn from data (compute the mean...). Only on **training** data.
- `transform` = apply what was learned. On **training and test** data.
- `fit_transform` = both steps in one. Only on **training** data. For the test set use `transform` alone.

## 4. When do we use it?

- Always, in supervised learning, before any model training and before any step that learns from the data.
- Notebook 15 will add **cross-validation**, which repeats the train/validate idea several times inside the training set. The test set stays untouched until the end.

## 5. Important terminology

| Term | Simple meaning |
|---|---|
| **Training set** | Rows used to learn |
| **Test set** | Rows held back to judge the final model |
| **`X_train, X_test, y_train, y_test`** | The standard names for the four pieces |
| **Hold-out** | Another name for keeping a test set aside |
| **Random split** | Rows are shuffled before cutting |
| **Stratified split** | A split that keeps the same percentage of each class in both parts |
| **`random_state` (seed)** | A fixed number that makes randomness repeatable |
| **Generalisation** | How well a model works on data it has not seen |
| **Data leakage** | Test (or future) information influencing training |
| **`fit` / `transform`** | Learn from data / apply what was learned |
| **Validation set** | A further part of training data used for tuning (we use cross-validation for this instead) |

## 6. Simple example

Ten customers' monthly bills: `20, 22, 25, 30, 35, 40, 45, 50, 55, 200`.

Suppose we want to scale values by subtracting the mean. We split 8 to train and 2 to test, and the test rows are `35` and `200`.

- **Wrong:** compute the mean from all ten (52.2). The test value 200 has pulled the average up, so the training data is shifted by information from the test set. This is leakage.
- **Right:** compute the mean from the 8 training values only (about 35.9), and use that same number to shift the test values. The test set had no say in it.

The difference looks small here, but with real models and many steps, leakage can make a score look several points better than it really is.

## 7. Connection to the ML workflow

```
Understand → Clean → EDA → Outliers → [5. Split] → Preprocess (fit on train) → Train → Evaluate on test → ...
                                          ^ you are here
```

Notebook 05 is the **gate**. Everything before it only looked at the data or applied fixed rules. Everything after it must respect the train/test wall:

- Encoding, scaling, imputing, feature selection, outlier fences: **fit on train only.**
- Model training: **train only.**
- Final score: **test, used once.**

We will build this safely in notebook 08 using a scikit-learn **Pipeline**, which does the fit-on-train-only rule automatically.

## 8. Common mistakes

1. **Preprocessing before splitting.** Scaling or imputing on the full dataset, then splitting. Classic leakage.
2. **Forgetting `stratify=y`** on imbalanced classes. You may end up with a test set that has too few examples of the minority class.
3. **Forgetting `random_state`.** Each run gives different results, so you cannot tell if a change helped.
4. **Peeking at the test set repeatedly.** If you tune the model by looking at test scores again and again, the test set becomes part of training. Use cross-validation for tuning, and the test set once.
5. **Using `fit_transform` on the test set.** Use `transform` only.
6. **Leaving target-related columns in X.** Always ask: "Would I have this information at prediction time?"
7. **Splitting rows that belong together** (for example, several rows from the same customer) into both sets. Not an issue in this dataset because each customer appears once.

## 9. Key takeaways

- Split into **train** (learn) and **test** (final exam) **before** any learning step.
- Use `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
- `stratify=y` keeps class proportions the same in both parts. `random_state` makes it repeatable.
- **Data leakage** makes results look better than reality. The fix: **fit on train only, transform both.**
- The test set is for **one final check**, not for tuning.
- Ask of every feature: "Will I know this at prediction time?"

---

**Next:** open the notebook for this note and run it cell by cell.
