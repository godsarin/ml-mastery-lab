# Note 04: Outlier Detection and Handling

> Read this note first. Then open `notebooks/01_supervised_churn/04_outliers.ipynb`.

---

## 1. What is it?

An **outlier** is a value that is very different from the rest of the values in a column. For example, if most customers pay between $20 and $120 per month and one record says $50,000, that value is an outlier.

Think of a class of students whose heights are between 150 cm and 190 cm, and one entry says 17 metres. Either someone made a typo, or a very unusual giraffe joined the class.

## 2. Why do we need it?

- Some algorithms are **sensitive** to extreme values. A single huge number can drag a model's line or distort distances and averages.
- Extreme values can be **errors** (typos, broken sensors), and errors should be fixed.
- Extreme values can also be **real and important** (a rare but genuine big customer), and deleting them would throw away information.

So outlier work is about **detecting** them, **understanding** them, and then **deciding** what to do. The decision is the hard part.

## 3. How does it work?

### Step 1: Detect

**Method A: the IQR rule** (most common, does not assume a bell shape)

1. Sort the values. Find Q1 (the value 25% of the way through) and Q3 (75% of the way through).
2. IQR = Q3 − Q1 (the spread of the middle half of the data).
3. Lower fence = Q1 − 1.5 × IQR. Upper fence = Q3 + 1.5 × IQR.
4. Anything outside the fences is flagged as an outlier.

**Method B: the z-score** (works best for bell-shaped data)

1. z = (value − mean) / standard deviation. It says "how many standard deviations from the average".
2. Flag values with |z| > 3.

**Method C: look at it.** A box plot or histogram often shows outliers immediately. Always look.

### Step 2: Understand

Ask *why* it is extreme:
- **Error?** (age = 250, negative price) → fix or remove.
- **Real but rare?** (a genuinely big customer) → usually keep.
- **A different group?** (a corporate account mixed in with households) → maybe model separately.

### Step 3: Handle

| Option | What it does | Use when |
|---|---|---|
| **Keep** | Do nothing | The values are real, or the model is not sensitive to them |
| **Remove rows** | Delete the extreme rows | They are clear errors, and there are few of them |
| **Cap (winsorize / clip)** | Replace values beyond the fences with the fence value | You want to limit the influence but keep the row |
| **Transform** | Apply a log or similar to squash large values | The data is strongly right-skewed |
| **Treat as missing** | Mark as NaN and impute | The value is impossible and the true value is unknown |

### Which models care?

| Sensitive to outliers | Fairly robust |
|---|---|
| Logistic Regression, Linear Regression, KNN, SVM, anything using distances or scaling | Decision Trees, Random Forest (they split on thresholds, so how extreme a value is does not matter) |

## 4. When do we use it?

After EDA, before preprocessing, when you see extreme values in box plots or histograms. Even when the check finds nothing, doing it is part of a good routine.

## 5. Important terminology

| Term | Simple meaning |
|---|---|
| **Outlier** | A value far from most others |
| **Quartile (Q1, Q3)** | The 25% and 75% positions after sorting |
| **IQR** | Interquartile range: Q3 − Q1 |
| **Fence** | The boundary beyond which values are flagged |
| **Z-score** | How many standard deviations a value is from the mean |
| **Standard deviation** | A typical distance of values from the average |
| **Winsorizing / capping / clipping** | Pulling extreme values in to a limit |
| **Log transform** | Replacing a value with its logarithm to shrink large numbers |
| **Skewed** | Lopsided distribution with a long tail |
| **Robust** | Not easily affected by outliers |

## 6. Simple example

Monthly spend of 9 customers: `20, 25, 22, 30, 28, 26, 24, 27, 900`

- Sorted, Q1 ≈ 24, Q3 ≈ 28 → IQR = 4.
- Fences: lower = 24 − 6 = 18, upper = 28 + 6 = 34.
- **900 is far above 34 → outlier.**
- The average with 900 is about 122, which describes nobody. Without it, about 25.
- Options: if 900 is a typo for 90, fix it. If it is a real heavy user, keep it, or cap it at 34 to limit its pull.

## 7. Connection to the ML workflow

```
Understand → Clean → EDA → [4. Outliers] → Split → Preprocess → Train → Evaluate → ...
                              ^ you are here
```

### Important: the leakage warning again

The fences (Q1, Q3) are **learned from data**. If you compute them on the whole dataset and then split, test data has influenced the training step. The proper way: compute the fences on the **training** data only, then apply those same fences to the test data. We will learn the split in notebook 05. In this notebook we only *detect*, and make no change to the real data, so nothing leaks.

Also: **never delete rows from the test set** just because they look strange. Real new data will also contain strange values, and your model must cope.

## 8. Common mistakes

1. **Deleting every outlier automatically.** Real, rare values are often the most interesting.
2. **Never looking.** A typo of one extra zero can ruin a linear model.
3. **Using z-scores on very skewed data.** The mean and standard deviation are themselves pulled by the outliers. The IQR rule is steadier.
4. **Calculating fences on all data before splitting.** This leaks information.
5. **Removing outliers from the test set.** This makes your test score look better than real life.
6. **Forgetting the model type.** Tree models do not need outlier handling for the sake of the model, and linear models and KNN do.
7. **Confusing "skewed" with "has outliers".** A long tail is not automatically a set of errors.

## 9. Key takeaways

- An outlier is a very unusual value, and it is **not automatically wrong**.
- Detect with the **IQR rule** (fences at Q1 − 1.5·IQR and Q3 + 1.5·IQR), the z-score, and charts.
- **Understand** before you act: error, rare-but-real, or different group?
- Options: keep, remove, cap, transform.
- Distance-based and linear models are sensitive, and tree models are robust.
- Calculate fences on **training data only**. Leave the test set alone.
- It is a valid and common result to find **no outliers**. Do not invent a problem.

---

**Next:** open the notebook for this note and run it cell by cell.
