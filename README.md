# 🎓 Student Loan Risk Prediction — HSLS Debt & Default Modeling

**Course:** INDENG 242A | UC Berkeley  
**Team:** JP Schuchter, Tonya Yermilova, Derrick Chun, Rameen Faisal  
**Live App:** [🔗 HSLS Student Loan Risk Explorer](https://ml242a-429797368432.us-west1.run.app)

---

## 📌 Overview

Most student loan planning tools tell you your monthly payment. None tell you your risk of defaulting or spending years underwater on debt. This project fills that gap.

Using the U.S. High School Longitudinal Study (HSLS) Postsecondary Education Transcript Study (PETS), we built predictive models for two long-run outcomes among federal student loan borrowers:

- **Default risk** — did the borrower ever default on a federal student loan?
- **Heavy debt burden** — does the borrower's debt-to-income (DTI) ratio exceed 0.8?

Models are deployed in an interactive Plotly Dash application where students can input their own profile and see predicted risk for both outcomes alongside what-if borrowing scenarios.

---

## 📊 Dataset

**Source:** HSLS 2009–16 Postsecondary Education Transcript Study (PETS) — a nationally representative cohort of 21,000+ U.S. students followed from high school into early adulthood.

| Model | Sample Size | Positive Rate |
|---|---|---|
| Default | ~6,700 borrowers | 2.6% (rare event) |
| Heavy Debt Burden (DTI > 0.8) | ~5,300 borrowers | 41.4% |

**Key engineered features:** intended field group (STEM, Health, Business, etc.), institution type (Public 4-year, Private non-profit, etc.), constructed annual earnings harmonized across pay units and survey waves, total federal debt from outstanding balances.

---

## ⚙️ Modeling Approach

Two supervised classification tasks, each with two feature sets:

**Full-feature model** — extended predictor set including demographics, SES, college characteristics, borrowing details, and earnings expectations.

**App-ready model** — restricted to 8 interpretable inputs a student can realistically provide before finalizing their borrowing decision: total intended borrowing, annual cost, intended field, institution type, family income, parent education, sex, and college selectivity.

**Algorithms compared:** Logistic Regression (L1 & L2), LDA, KNN, SVM, Random Forest, Gradient Boosting

**Evaluation:** 5-fold stratified cross-validation — ROC AUC, accuracy, precision, recall, F1, confusion matrices, threshold tuning for imbalanced outcomes.

---

## 📈 Results

| Outcome | Model | ROC AUC |
|---|---|---|
| Default | Full-feature Logistic Regression | ~0.83 |
| Heavy Debt Burden | Full-feature Logistic Regression | ~0.91 |
| Heavy Debt Burden | Gradient Boosting | ~0.91–0.92 |
| Default | App-ready Logistic Regression | ~0.67–0.68 |
| Heavy Debt Burden | App-ready Logistic Regression | ~0.67 |

Heavy debt burden is substantially more predictable than default due to its more balanced class distribution. Default modeling is limited by severe class imbalance (2.6% positive rate) — models are most useful as broad risk signals for flagging very high-risk borrowers rather than precise classifiers.

Key coefficient findings: low family income and parent education drive default risk; higher total borrowing and annual program costs drive heavy debt burden. Coefficients differ meaningfully across the two tasks, confirming they capture distinct financial outcomes.

---

## 🖥️ Live Dashboard — HSLS Student Loan Risk Explorer

**[→ Open the App](https://ml242a-429797368432.us-west1.run.app)**

The Plotly Dash app includes six pages:

- **Risk Explorer** — input your profile via sliders/dropdowns and get predicted default and high-debt probabilities, percentile ranks, and risk vs. HSLS baseline comparisons
- **Model Diagnostics** — side-by-side performance summaries, calibration curves, ROC and precision-recall curves, threshold tradeoff visualizations
- **PCA Profiles** — borrower profiles embedded in lower-dimensional space, colored by cluster
- **Exploratory Patterns** — joint distributions, income by group, earnings by group, institution type breakdowns
- **Feature Effects** — top app-ready logistic coefficients for each outcome with log-odds interpretation
- **Data** — dataset summary, sample rows, cleaning notes, and responsible use guidelines

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

**Key concepts:** Binary classification, class imbalance handling, threshold tuning, logistic regression interpretation, gradient boosting, PCA, cross-validation, Plotly Dash deployment

---

## ⚠️ Limitations

- Default is rare — even strong AUC can mask low precision and recall at standard thresholds
- Earnings are constructed from multiple survey waves, introducing measurement error into DTI
- HSLS reflects a 2009 cohort; tuition, labor markets, and repayment options have changed since
- Features like family income and institution type correlate with protected attributes — full fairness auditing would be required before any real deployment

This is a proof-of-concept academic project. Do not use for lending, eligibility, or high-stakes decisions.

---

## 👩‍💻 Author

**Rameen Faisal** — Master of Analytics, UC Berkeley  
[LinkedIn](https://www.linkedin.com/in/rameen-faisal/) · [rameen@berkeley.edu](mailto:rameen@berkeley.edu)
