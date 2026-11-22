# Note 03: Exploratory Data Analysis (EDA) and Visualization

> Read this note first. Then open `notebooks/01_supervised_churn/03_eda.ipynb`.

---

## 1. What is it?

**Exploratory Data Analysis (EDA)** means investigating your data with summary numbers and charts to discover patterns, relationships and surprises, *before* building any model.

Think of a detective at a crime scene. The detective looks around, collects clues and forms ideas about what happened. Only then do they name a suspect. EDA is that stage for your data.

## 2. Why do we need it?

- It tells you **which features seem to matter** for the target.
- It reveals **problems** that cleaning missed: skewed values, extreme values (outliers), features that overlap.
- It guides decisions about **scaling, encoding and feature engineering**.
- It lets you **check the result later**: if a model says "contract type matters most", EDA should already have shown you this. If not, something may be wrong.
- Charts communicate findings to people who do not read code.

## 3. How does it work?

EDA moves from simple to detailed:

1. **Univariate analysis** (one variable): what does each column look like by itself?
   - Numbers → histogram (how values are spread).
   - Categories → bar chart (how many in each group).
2. **Bivariate analysis** (two variables): how does each feature relate to the target?
   - Category vs. target → *churn rate per group* (the percentage that churned in each group).
   - Number vs. target → box plot or histogram split by target.
3. **Multivariate analysis** (many variables): how do the features relate to each other?
   - Numbers vs. numbers → **correlation matrix** shown as a heatmap.

### Which chart for which question?

| Question | Chart |
|---|---|
| How are the values of one number spread? | Histogram |
| How many in each category? | Bar chart |
| Does a number differ between two groups? | Box plot |
| Does a category change the target rate? | Bar chart of rates per group |
| How strongly are numbers linked? | Correlation heatmap |

## 4. When do we use it?

After cleaning and before modelling. Do EDA on the **training data** in a strict project, to avoid your decisions being influenced by test data. In this learning project we explore the cleaned full dataset to understand the problem, and make no decisions that use statistics from the test set. (Notebook 05 explains the split.)

## 5. Important terminology

| Term | Simple meaning |
|---|---|
| **Distribution** | How the values of a column are spread out |
| **Histogram** | A chart that counts how many values fall in each range ("bin") |
| **Skewed** | A distribution with a long tail on one side (for example, most people earn little, a few earn a lot) |
| **Median** | The middle value. Half the values are below it, half above |
| **Quartiles / IQR** | The 25% and 75% positions. IQR = 75% value − 25% value |
| **Box plot** | A chart showing the median, the middle 50% of values (the box) and extreme values (dots) |
| **Churn rate** | Percentage of customers in a group who churned |
| **Correlation** | A number from −1 to +1 for how two numeric columns move together |
| **Positive correlation** | When one goes up, the other tends to go up |
| **Negative correlation** | When one goes up, the other tends to go down |
| **Multicollinearity** | Two or more features that carry nearly the same information |
| **Univariate / bivariate** | Looking at one variable / two variables |

### Reading correlation

- **+1**: perfect rise together. **−1**: perfect opposite. **0**: no straight-line link.
- Rough guide: below 0.3 is weak, 0.3 to 0.7 is moderate, above 0.7 is strong.
- **Correlation is not causation.** Ice cream sales and sunburn are correlated because both rise in summer. Ice cream does not cause sunburn.
- Correlation only measures *straight-line* relationships between numbers. It does not capture every kind of link.

## 6. Simple example

Ten customers, with the number of months they stayed:

`Stayed: 30, 40, 36, 50, 28`  `Left: 2, 5, 3, 8, 4`

- The average for those who left is about 4 months. For those who stayed it is about 37.
- A chart of both groups would show two clearly separate clusters.
- **Insight:** *tenure* strongly separates the two groups. A model will find it useful.

That is the whole spirit of EDA: compare groups and look for differences.

## 7. Connection to the ML workflow

```
Understand → Clean → [3. EDA] → Outliers → Split → Preprocess → Train → Evaluate → ...
                        ^ you are here
```

What EDA gives the next steps:
- Skewed or extreme values → notebook 04 (outliers).
- Features on very different scales → notebook 06 (scaling).
- Overlapping features → notebook 07 (feature selection).
- A baseline expectation of which features matter → useful when we interpret the models.

## 8. Common mistakes

1. **Making charts without a question.** Decide what you want to learn first.
2. **Confusing correlation with causation.** "Customers with fiber optic churn more" does not prove fiber optic *causes* churn.
3. **Reading too much from a small group.** A 100% churn rate in a group of 3 people means nothing. Check group sizes.
4. **Using only averages.** The mean hides the spread. Look at distributions.
5. **Correlating categories coded as numbers.** `SeniorCitizen` is 0/1 but is really a category, so treat it as one.
6. **Using statistics from the test set to make decisions.** This leaks information (see notebook 05).
7. **Unreadable charts.** Always add a title and axis labels.

## 9. Key takeaways

- EDA goes **one variable → two variables → many variables**.
- Always compare the **target rate across groups** (churn rate per category).
- Use the right chart for the question.
- **Correlation is not causation**, and only measures straight-line links between numbers.
- Strongly overlapping features (multicollinearity) are worth noting.
- EDA creates hypotheses ("short-tenure customers churn more"). Models then test them.

---

**Next:** open the notebook for this note and run it cell by cell.
