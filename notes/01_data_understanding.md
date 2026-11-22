# Note 01: Understanding Your Data

> Read this note first. Then open `notebooks/01_supervised_churn/01_data_understanding.ipynb`.

---

## 1. What is it?

Before a machine learning model can learn anything, **you** must understand the data. This step means answering simple questions:

- How many rows and columns are there?
- What does each column mean?
- Which column are we trying to predict?
- What type of data is in each column (numbers, text, dates)?

Think of it like a doctor reading a patient's file before deciding on a treatment. No doctor prescribes medicine before reading the file.

## 2. Why do we need it?

- A model only learns from the data you give it. Wrong or misunderstood data gives wrong predictions, however clever the algorithm.
- Most of a real ML project's time (often 60-80%) goes into data work, not model work.
- Spotting problems early (a column stored as the wrong type, an ID column, a heavily unbalanced target) saves hours later.

## 3. How does it work?

A standard first look follows five steps:

1. **Load** the data into a table (a Pandas `DataFrame`).
2. **Look** at the first few rows (`head`).
3. **Check the size** (rows × columns, called the *shape*).
4. **Check each column's type** and how many values are filled in (`info`).
5. **Identify the target** and look at how its values are spread.

## 4. When do we use it?

Every time you get a new dataset: a CSV from a company, a download from Kaggle, an export from a database. Always. It is the first step of every ML project.

## 5. Important terminology

| Term | Simple meaning | Churn example |
|---|---|---|
| **Dataset** | The whole table of data | All 7,043 customers |
| **Row / sample / observation** | One example | One customer |
| **Column** | One piece of information about each example | `tenure`, `Contract` |
| **Feature** (also *input*, *independent variable*, *X*) | A column the model uses to make a prediction | `tenure`, `MonthlyCharges` |
| **Target** (also *label*, *output*, *dependent variable*, *y*) | The column we want to predict | `Churn` |
| **Supervised learning** | Learning from examples where the right answer is known | We know which customers left |
| **Classification** | Predicting a category | Will leave: Yes or No |
| **Regression** | Predicting a number | House price |
| **Numerical feature** | A column of numbers you can do maths on | `MonthlyCharges` |
| **Categorical feature** | A column with a limited set of groups | `Contract`: Month-to-month, One year, Two year |
| **Identifier (ID)** | A unique label for each row, with no real meaning | `customerID` |
| **Data type (dtype)** | How Pandas stores a column (`int64`, `float64`, text) | |
| **Class imbalance** | One target value is much more common than the other | 73% stay, 26% leave |

## 6. Simple example

A tiny dataset:

| customerID | tenure (months) | Contract | Churn |
|---|---|---|---|
| A1 | 2 | Month-to-month | Yes |
| A2 | 60 | Two year | No |
| A3 | 5 | Month-to-month | Yes |

- **Rows:** 3 customers.
- **Features:** `tenure`, `Contract`. (Not `customerID`, which is just a name tag.)
- **Target:** `Churn`.
- **Pattern a model might learn:** new customers on month-to-month contracts tend to leave.

The model's job is to find patterns like this in thousands of rows, then apply them to new customers whose `Churn` is unknown.

## 7. Connection to the ML workflow

```
[1. Understand data] → Clean → EDA → Split → Preprocess → Train → Evaluate → Tune → Save → Predict
        ^ you are here
```

Everything that follows depends on this step. The decisions you make here (which column is the target, which columns to drop, which columns are numerical or categorical) decide how you clean, encode and scale the data later.

## 8. Common mistakes

1. **Skipping the look.** Jumping straight to model training without reading the data.
2. **Treating IDs as features.** `customerID` is unique for every row, so it carries no pattern. Models can use it to memorise rows instead of learning.
3. **Trusting the type Pandas guesses.** A number column can be stored as text if even one value is a blank space. Always check `dtypes`.
4. **Confusing "no missing values" with "clean".** Missing values can hide as blanks `" "`, `"?"` or `"N/A"`, which Pandas does not count as missing.
5. **Ignoring the target balance.** If 95% of the rows are "No", a model that always says "No" gets 95% accuracy and is useless.
6. **Mixing up features and target.** Leaving the target inside the features lets the model cheat (this is called *data leakage*, covered in a later note).

## 9. Key takeaways

- Understand the data before modelling. Always.
- **Features (X)** are the inputs. The **target (y)** is what we predict.
- Churn prediction is **classification**: the answer is a category (Yes/No).
- Check the **shape**, the **data types** and the **target distribution** first.
- Drop **identifier** columns from the features.
- If the target is **imbalanced**, accuracy alone can mislead. We will use better metrics later.
- Data can look fine and still hide problems (we will find one in this dataset, and fix it in Note 02).

---

**Next:** open the notebook for this note and run it cell by cell.
