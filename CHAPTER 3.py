from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:1234@localhost:5432/bank_marketing")
import pandas as pd
bank = pd.read_sql("SELECT * FROM bank_ml", engine)

#this time I will use 10 000 rows but with new random_state 99 = new data
ratio =10000 / len(bank)
bank_small = (
    bank.groupby("y", group_keys = False)
        .sample(frac= ratio, random_state= 99)
        .reset_index(drop = True)
    )

bank_cluster = bank_small[["age", "job", "marital", "education", "default", "housing", "loan"]]

#clustering preprocess - I am using k-prototypes - many categorical and few numeric
cat_cols1 = bank_cluster.select_dtypes(include=["object","string","category"]).columns
num_cols1 = bank_cluster.select_dtypes(include=["int64","float64","int32","float32"]).columns
bank_cluster[cat_cols1] = bank_cluster[cat_cols1].astype(str)
bank_cluster[num_cols1] = bank_cluster[num_cols1].astype(float)
bank_cluster.info()
#it was already in correct datatype but for future I want to keep it here

#clustering
cat_id = [bank_cluster.columns.get_loc(c) for c in cat_cols1] #ids of cat. columns
x_np = bank_cluster.to_numpy() #we need numpy array for this clustering
from kmodes.kprototypes import KPrototypes
kproto = KPrototypes(n_clusters=8, init="Cao", n_init=5, random_state=9) #init ="Cao" good for categorical features, n_init = number of initialization
bank_clusters_done = kproto.fit_predict(x_np, categorical=cat_id) #for each customer will be assigned cluster
bank_cluster["cluster"] = bank_clusters_done + 1 #+1 because usually first cluster is 0 and I want it to be 1
print(bank_cluster["cluster"].value_counts())

#now lets test which cluster has the biggest predicted y=1 ratio  
#firstly we need to add economics features and make x and y variables
bank_eco = bank_small[["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]]
bank_cluster_eco = pd.concat([bank_cluster.reset_index(drop=True), bank_eco.reset_index(drop=True)], axis=1) #add economics futures for model prediction
bank_cluster_eco = bank_cluster_eco.loc[:, ~bank_cluster_eco.columns.duplicated()]
x = bank_cluster_eco
x = x.drop(columns=["cluster"])
x.info()
y = bank_small["y"] #for validation
#now import model from previous chapter
import joblib
model_demographic_eco = joblib.load("bank_predict_y.joblib")
bank_cluster_y_predict = model_demographic_eco.predict_proba(x) [:, 1]
y_predict = (bank_cluster_y_predict>=0.45).astype(int)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y, y_predict))
print(classification_report(y, y_predict))

#results are fine, now add dataset next to cluster col y_predict and check y ratios for every cluster 
bank_cluster["y_predict"] = y_predict
print(bank_cluster)
clients_by_cluster = {}
for c in bank_cluster["cluster"].unique(): #subset for each cluster
    clients_by_cluster[c] = bank_cluster[bank_cluster["cluster"] == c].copy()
print(clients_by_cluster[4])

#y ratio - top 3 clusters for marketing targeting
best_clusters = (
    bank_cluster
      .groupby("cluster")["y_predict"]
      .agg(ratio="mean", n="count")
      .sort_values("ratio", ascending=False)
      .head(3)
)
print(best_clusters)

""" 
all clusters
Cluster     ratio     n
6        0.985816   141 
7        0.347601   771 
5        0.236760  1284 
1        0.221477  1192 
8        0.179001  2162 
4        0.123550  1983
3        0.118397  1098
2        0.116143  1369
"""
#now we are going to check what makes customers in the top 3 clusters similar

#I need devide bank_cluster cols into numerical and categorical cols
#I have done it before... I have correct cols in previous variable cat_cols1 so I can use it again 
#but in num_cols1 there are cluster and y_predict, we do not want them there, we want just Age
print(bank_cluster.info())
num_cols1 = bank_cluster.select_dtypes(include=["float64"]).columns
print(cat_cols1)
print(num_cols1)
#Numeric (age)
cluster_age_median = bank_cluster.groupby("cluster")[num_cols1].median()
print(cluster_age_median)
#Categorical - top 1 category
for c in cat_cols1:
    print(f"\n=== {c} ===");print(bank_cluster.groupby("cluster")[c].value_counts(normalize=True).groupby(level=0).head(1)) #normalize = True => shows ratio, without normalize = n

""" Best three segments for targeting
Cluster     Age     Job             Marital         Education           Default         Housing         Loan

Cluster 6:  73      retired         married         basic.4y            no              yes             no
Cluster 7:  26      blue-collar     single          high.school         no              yes             no
Cluster 5:  30      admin           single          university.degree   no              yes             no
"""

results = pd.DataFrame({
    "Cluster":[6, 7, 5, 1, 2, 3, 4, 8],
    "Age":[73, 26, 30, 56, 48, 33, 42, 37],
    "Job":["retired", "blue-collar", "admin", "admin", "blue-collar", "blue-collar","admin","admin"],
    "Marital":["married", "single", "single", "married", "married", "married", "married", "married"],
    "Education":["basic.4y", "high.school", "university.degree", "university.degree", "university.degree","high.school", "university.degree","university.degree"],
    "Default":["no", "no", "no","no","no","no","no","no"],
    "Housing":["yes", "yes", "yes", "yes","yes","no", "no","yes"],
    "Loan":["no", "no", "no", "no", "no", "no", "no", "no"],
    "y_ratio": [0.985816, 0.347601, 0.236760, 0.221477, 0.116143, 0.118397, 0.123550, 0.179001],
    "n": [141, 771, 1284, 1192, 1369, 1098, 1983, 2162],
    "model_precision": [0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]
}).sort_values("y_ratio", ascending=False)
print(results.info())
#number of people with y=1
results["n_of_y1"] = round(results["n"]*results["y_ratio"])
#number of customers who actually open a term deposit - adjusted for model accuracy
results["n_customers"] = round(results["n_of_y1"] * results["model_precision"])
#conversion rate - number of people who opened a term deposit/number of people approached (%)
results["conversion_rate"] = (results["n_customers"]/results["n"])*100
print(results)

coversion_rate_bank = (bank_small["y"].sum()/len(bank_small))*100
print(coversion_rate_bank)

"""
Clusters 5, 6 and 7 show a better conversion rate than the dataset average, with Cluster 1 being about 25% more effective. 
Even though the sample size is smaller, it's a great target for banks without the budget for mass-marketing. 
Keep in mind, though, that the dataset is quite imbalanced (mostly zeros in the target variable) and we only 
trained the model on 8% of the data to save on processing power.
"""

bank_cluster["y_real"] = bank_small["y"] #add actuall y for reporting
bank_cluster.to_sql(
    "bank_cluster",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)
results.to_sql(
    "results",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)

#Next chapter: Power BI reports 
