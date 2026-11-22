# Note 02: Data Cleaning

> Read this note first. Then open `notebooks/01_supervised_churn/02_data_cleaning.ipynb`.

---

## 1. What is it?

Data cleaning means finding and fixing problems in your data so that a model can use it correctly. Real data is messy: values are missing, numbers are stored as text, rows are repeated, and the same category is spelled in different ways.

Think of washing vegetables before cooking. The recipe (the model) is only as good as the ingredients (the data).

## 2. Why do we need it?

- Most algorithms **crash or give wrong results** if there are missing values or text where numbers should be.
- Duplicated rows can make the model count the same example twice.
- Wrong data types stop you from doing maths (you cannot average text).
- "Garbage in, garbage out": a perfect algorithm on dirty data still gives bad predictions.

## 3. How does it work?

A cleaning pass has four checks:

1. **Missing values**: find them (including *hidden* ones like `" "`), then decide what to do.
2. **Data types**: make sure numbers are numbers and categories are categories.
3. **Duplicates**: find repeated rows and decide whether they are real mistakes.
4. **Inconsistent values**: the same thing written in different ways (`"Yes"`, `"yes"`, `"Y"`).

### Handling missing values: your options

| Option | What it means | Use when |
|---|---|---|
| **Drop the rows** | Delete rows with missing values | Very few rows are affected, and they are not special |
| **Drop the column** | Delete the whole column | Most of the column (say 50%+) is missing |
| **Fill with a statistic** (*imputation*) | Mean, median or most frequent value | Values are missing at random |
| **Fill using domain knowledge** | A value that makes real-world sense | You understand *why* it is missing |
| **Add a "was missing" flag** | A new column saying the value was missing | The fact that it is missing might carry meaning |

**Mean vs. median:** the mean is pulled by extreme values (one millionaire raises the average income a lot). The median is the middle value and is more robust. When unsure, the median is the safer choice for numbers, and the most frequent value (*mode*) for categories.

## 4. When do we use it?

Right after you understand the data (Note 01) and before any analysis or modelling. You will also meet cleaning again inside pipelines (Note 08), where the statistic-based filling is done safely.

## 5. Important terminology

| Term | Simple meaning |
|---|---|
| **Missing value** | An empty cell. Pandas shows it as `NaN` ("not a number") |
| **Hidden missing value** | A blank or placeholder (`" "`, `"?"`, `"N/A"`, `-999`) that Pandas does *not* treat as missing |
| **Imputation** | Filling missing values with a sensible replacement |
| **Duplicate row** | A row that repeats another row |
| **Data type conversion** | Changing how a column is stored (text → number) |
| **`NaN`** | Pandas' marker for "no value here" |
| **Coercion** | Forcing a conversion, turning anything that fails into `NaN` |
| **Domain knowledge** | Knowing how the real-world business works |
| **Data leakage** | Information from the test data accidentally influencing training (explained below) |

## 6. Simple example

| name | age | city |
|---|---|---|
| Ana | 25 | Paris |
| Ben | `NaN` | Rome |
| Ben | `NaN` | Rome |
| Cara | 31 | `" "` |

Problems here:
- Ben's age is **missing**. Options: fill with the median age of 28 (the middle of 25 and 31).
- Rows 2 and 3 are **identical**. If it is the same person recorded twice, drop one.
- Cara's city is a **hidden missing value**: it looks filled but is just a blank space.

## 7. Connection to the ML workflow

```
Understand → [2. Clean] → EDA → Split → Preprocess → Train → Evaluate → Tune → Save → Predict
                ^ you are here
```

### Important: the leakage warning

There are two kinds of cleaning:

1. **Rule-based fixes** that need no learning from data: fixing a data type, removing a blank space, filling a value using a fixed real-world rule. These are safe to do on the full dataset *before* splitting.
2. **Statistic-based fixes** such as "fill with the median": the median is *learned from the data*. If you compute it using the whole dataset, information from the future test set leaks into training. These must be fitted **only on the training data**. We will do this inside a pipeline (Notes 05 and 08).

In this notebook, our `TotalCharges` fix is a *rule* (a customer who has not been billed has paid 0), so it is safe to do now.

## 8. Common mistakes

1. **Believing `isna()` finds everything.** It does not find blanks, `"?"` or `-999`. Look at the values themselves.
2. **Dropping rows without thinking.** You lose data, and the missing rows may be different from the rest (for example all new customers).
3. **Filling with the mean on the full dataset before splitting.** This is data leakage.
4. **Blindly dropping all duplicates.** Two customers can legitimately have the same answers. Duplicates only matter if they are the *same real record* repeated.
5. **Not checking the result.** Always re-check: do the types look right now? Is any `NaN` left?
6. **Overwriting the original data.** Keep the raw file untouched and save the cleaned data as a new file.

## 9. Key takeaways

- Check four things: missing values, data types, duplicates, inconsistent values.
- Missing values can be **hidden**. Look inside the data, not only at `isna()`.
- Choose a fix by understanding **why** the value is missing, not by habit.
- Use **domain knowledge** when you have it.
- Rule-based fixes are safe before the split. **Statistic-based fills belong in the training pipeline.**
- Never edit the raw file. Save cleaned data separately.

---

**Next:** open the notebook for this note and run it cell by cell.
