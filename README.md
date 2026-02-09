## **Introduction and project context**

Banks, like any other company, use marketing campaigns to sell financial products (e.g., term deposits, loans). In practice, a big part of the outreach doesn’t lead to anything – campaigns cost time and money, but don’t always bring the expected revenue growth. That’s why I used **data** to help.

This project works with a dataset from the **UCI Machine Learning Repository**, which contains **41,000+ client records** from a Portuguese bank. The data covers **2008–2010** and includes **macroeconomic indicators**, **customer demographics**, and **customer behavior**.  
**Dataset link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## **Main goal**

- Build a **predictive model** that estimates whether a client will subscribe to a **term deposit** *before the marketing campaign*.
- Identify the **most important variables** that influence term deposit subscription.
- Design a suitable **customer segmentation (clustering)** to target campaigns to specific groups more effectively.

---

## **Partial goals**

- Get the data, modify it, enrich it, clean it, and load it into a **database** for further use.
- Run **permutation importance** and select the most relevant variables.
- Build, evaluate, and compare **multiple predictive models**.
- Visualize and interpret results in **Power BI** and use them to gain a competitive advantage (**campaign recommendations**).

---

## **Used skills**

**PostgreSQL**, **SQL**, **Python**, **Power BI**, **Pandas**, **Scikit-learn**, **SQLAlchemy**, **NumPy**, **K-Prototypes**, **Logistic Regression**, **OneHotEncoder**, **ColumnTransformer**, **Permutation Importance**, **RandomForestClassifier**, **GridSearchCV**, **Model validation**, **Pipelines**, **DAX**, **Joblib** and more...

---

## **Chapter 1 - Data preparation**

- Prepared the data so it can be used for **both modeling and reporting**.
- Loaded the UCI Bank Marketing dataset and created a **backup** to track changes.
- Simulated **10% missing values** in columns `age`, `campaign`, `loan` and saved masks for validation.
- Filled `age` and `campaign` using a **KNN imputer** (after scaling) and checked quality using **MAE** and **RMSE**.
- Recovered `loan` as a **classification task** using **logistic regression** with preprocessing (scaling + one-hot encoding) and evaluated it with **confusion matrix + accuracy**.
- Removed duplicates and created **2 outputs**:
  - **`bank_report`** for Power BI (added feature **`new_customer`** from `pdays`)
  - **`bank_ml`** for the ML pipeline (categories filled with `"unknown"`, target `y` mapped to **0/1**)
- Loaded both tables into **PostgreSQL** (`bank_report`, `bank_ml`) using **SQLAlchemy**.

---

## **Chapter 2 - Predictive models + selection of the most important variables**

- Loaded cleaned data from PostgreSQL (`bank_ml`) and, for performance reasons, used **stratified sampling (~5,000 rows)** to keep the same class ratio in `y`.
- Created **2 sets of inputs**:
  - **Personal (pre-campaign):** `age, job, marital, education, default, housing, loan`
  - **Economic:** `emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed`
- Built a **RandomForest classifier** in a **Pipeline** (OneHotEncoder for categorical features) and tuned hyperparameters with **GridSearchCV** (optimized for **F1** and **accuracy**).
- Evaluated results using **confusion matrix + classification report** and adjusted the **decision threshold** (goal: capture the “yes” class better in imbalanced data).
- Identified key variables using **permutation_importance** – in the “personal” model the top factors were mainly **job, age, education**.
- Compared a model **without macro features** vs. **with macro features** – the model with economic variables performed better, so I saved the final version as **`bank_predict_y.joblib`** for further use (Power BI / scoring).
- Next step: **clustering** for segmentation and better targeting.

---

## **Chapter 3 - Customer segmentation**

- Loaded data from PostgreSQL (`bank_ml`) and created a stratified sample of **~10,000 rows** (new seed) to keep the class ratio in `y`.
- Used **K-Prototypes clustering** (mix of numeric + categorical features), using only “personal” inputs: `age, job, marital, education, default, housing, loan`.  
  ✅ Result: **8 clusters**
- For each cluster I applied the saved predictive model (from Chapter 2) and computed:
  - **purchase probability** (`y_predict`)
  - **share of “buyers”** inside each cluster
- Selected the **Top 3 clusters** based on the highest share of `y_predict=1` and created a short profile (median age + most common categories) to make it clear **who to target**.
- Built a reporting summary with: **ratio, segment size (n), estimated conversions, conversion rate** (vs dataset average).
- Saved outputs back to the database:
  - **`bank_cluster`** (each client + cluster + prediction + real `y`)
  - **`results`** (aggregated results by clusters)

---

## **Chapter 4 - Power BI**

- Connected Power BI directly to **PostgreSQL** and imported tables **`bank_report`**, **`bank_cluster`**, **`results`**.
- In **Power Query** I did quick cleaning and prep for visuals:
  - column types (e.g., **whole number** where it makes sense),
  - unified values in **`day_of_week`**,
  - created **age ranges** using **Column from Examples** (e.g., 32–41, 50–59 …).
- In **DAX** I added a helper column **`day_in_order`** using **SWITCH**, so weekdays show correctly (Mon→Sun) and can be sorted properly on chart axes.
- Created a relationship **`results → bank_cluster`** (1:* via `cluster`, cross-filter as needed), so aggregated segment metrics work together with client-level filtering.
- Built reports (charts + KPIs) and improved usability:
  - custom **Measures** in DAX,
  - **Edit interactions** between visuals,
  - **Buttons + page navigation**,
  - basic styling (readability, number formatting, consistent layout).
- Reports:
  - **Demographics Data** – visuals to better understand what kinds of customers the bank has
  - **Marketing Data** – visuals of marketing campaign outcomes
  - **Predictions Outcome** – visuals from chapters 2–3 + **Drill Through** (click a cluster to open detail: `n` in cluster, % with TD, conversion rate, estimated “guaranteed” deposits + segment profile job/marital, age range, housing/loan)

## **Demographics Data**

![Demographic Preview](demographic%20data.png)

## **Marketing Data**

![Marketing Preview](marketing%20data.png)

## **Predictions Outcome**

![Marketing Preview](predictions%20results.png)

## **Drill Through Cluster 6**

![Marketing Preview](Drill%20through.png)

---

## **Conclusion**

- Built an end-to-end pipeline: **CSV → cleaning/imputation → PostgreSQL → ML model → clustering → Power BI reports**.
- Created two data layers: **`bank_report`** (reporting) and **`bank_ml`** (modeling), plus outputs **`bank_cluster`** and **`results`** for segmentation.
- Tested data prep quality (imputation validation) and built a tuned **Random Forest** model with interpretation via **permutation importance**.
- Segmented customers using **K-Prototypes** (8 clusters) and identified **TOP segments** based on predicted term deposit subscription share.
- Turned results into actionable recommendations in **Power BI**: management dashboards + a “data science” page + drill-through to cluster detail.
- **Clusters 6, 7 and 5** had a higher conversion rate than the dataset average, with **Cluster 6 ~25% more effective** – a great target especially for banks without the budget for mass marketing (even though it’s a smaller segment).
- The dataset is **imbalanced** (mostly `y=0`) and, for performance reasons, I trained the model on only **~8% of the data**, so results should be seen as a **practical prototype** that could be improved with full training.
