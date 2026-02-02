#Get Data from database
from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:1234@localhost:5432/bank_marketing")
import pandas as pd
bank = pd.read_sql("SELECT * FROM bank_ml", engine)

#there are too many rows for my PC, I want to keep like 5k rows but leave the same ratio of groups in Y
ratio =5000 / len(bank)
bank_small = (
    bank.groupby("y", group_keys = False)
        .sample(frac= ratio, random_state= 9)
        .reset_index(drop = True)
    )
bank_small.columns
bank_personal = bank_small[["age", "job", "marital", "education", "default", "housing", "loan", "y"]]
bank_eco = bank_small[["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]]

#Has the client subscribed a term deposit (no campaign, no eco) - randomForest #1

#datasplit for train/test no eco, no camp
bank_personal.info()
x1=bank_personal[["age","job", "marital", "education", "default", "housing", "loan"]]
y1=bank_personal["y"]
from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.3, random_state=9, stratify=y1) #stratify when there is unbalance between classes in y

#OneHotEncode
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
num_cols1 = x1.select_dtypes(include=["int64", "float64"]).columns
cat_cols1 = x1.select_dtypes(include=["string","category" , "object"]).columns
preprocess1 = ColumnTransformer([
        ("num", "passthrough", num_cols1), #dont need Scaler in RF
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols1),#handle_unknown="ignore" if there is unknown category in test set, it won't crash
    ]
)

#Pipeline
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
pipe1 = Pipeline([
    ("prep", preprocess1),
    ("rf", RFC (random_state=9, n_jobs=1, class_weight="balanced_subsample")) #class_weight="balanced_subsample" - for unbalancey y variables
])

#forest parameters test
parmameter_test_1 = {
    "rf__n_estimators" : [100, 300, 500], #rf__ because of pipe
    "rf__max_depth" : [None, 5],
    "rf__min_samples_leaf": [5, 10],
    "rf__min_samples_split" : [2, 5, 10],
    "rf__max_features" : ["sqrt", 0.8] #OneHot will change number of features, this will prevent it "sqrt" - sqrt of count of variables, 0.9 - 90% of variables....
}

#setting up grid
from sklearn.model_selection import GridSearchCV as GSC
grid_personal = GSC(
    estimator = pipe1,
    cv=3,
    param_grid=parmameter_test_1,
    scoring='f1',
    n_jobs=6 #6 cores
)
grid_personal.fit(x_train1, y_train1)

#test 1
best_personal1 = grid_personal.best_estimator_
predict1_test = best_personal1.predict_proba(x_test1) [:, 1]
predict1_test_TH = (predict1_test >= 0.45).astype(int) #threshold
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test1, predict1_test_TH))
print(confusion_matrix(y_test1, predict1_test_TH))
print(grid_personal.best_params_)
"""
              precision    recall  f1-score   support

           0       0.95      0.39      0.55      1331
           1       0.15      0.84      0.25       169

    accuracy                           0.44      1500
   macro avg       0.55      0.61      0.40      1500
weighted avg       0.86      0.44      0.52      1500
"""
#This model has just 44 % accuracy but recall is 39 % (0), 84 % (1), precision for 95% (0), 15 % (1) 
#While accuracy is at 44%, the model's predictive power is solid—95% for non-buyers and 84% for buyers. To get the volume we need,
#we must increase our reach and run a more massive advertising campaign -

#best accuracy value 2
grid_personal2 = GSC(
    estimator = pipe1,
    cv=3,
    param_grid=parmameter_test_1,
    scoring='accuracy',
    n_jobs=6 #6 cores
)
grid_personal2.fit(x_train1, y_train1)
#test 2
best_personal2 = grid_personal2.best_estimator_
predict2_test = best_personal2.predict_proba(x_test1) [:, 1]
predict1_test_TH2 = (predict2_test >= 0.6).astype(int) #threshold
print(classification_report(y_test1, predict1_test_TH2))
print(confusion_matrix(y_test1, predict1_test_TH2))
print(grid_personal.best_params_)
"""
              precision    recall  f1-score   support

           0       0.90      0.93      0.91      1331
           1       0.24      0.17      0.20       169

    accuracy                           0.85      1500
   macro avg       0.57      0.55      0.55      1500
weighted avg       0.82      0.85      0.83      1500
"""
#best columns 2
from sklearn.inspection import permutation_importance 
#permutation importance
best_cols1_test = permutation_importance( #corrupting value for cols. Importance_mean,  importance_std more = stronger
    best_personal2, x_test1, y_test1,
    n_repeats=10,
    random_state=9,
    scoring="f1"
)
best_cols = pd.DataFrame({ #new dataframe
    "feature": x_test1.columns, #cols from bank_personal
    "importance_mean": best_cols1_test.importances_mean,
    "importance_std": best_cols1_test.importances_std,
}).sort_values("importance_mean", ascending=False)
print(best_cols)

#most important cols : Job, Age, education
"""
feature  importance_mean  importance_std
job         0.043124        0.017024
age         0.038718        0.017596
education   0.021632        0.014960
default     0.011029        0.008513
marital     0.003204        0.011382
loan       -0.001395        0.005852
housing    -0.003670        0.014574
"""
#stats/economics variables model3 - comparison with model 1
xy3 = pd.concat([bank_personal.reset_index(drop=True), bank_eco.reset_index(drop=True)], axis=1) #axis=1 -adding horizontally axis=0 adding vertically
xy3 = xy3.loc[:, ~xy3.columns.duplicated()]
x3 = xy3.drop(columns=["y"]) #dropping y 
y3 = xy3["y"]
x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.3, random_state=9, stratify=y3)
num_cols3 = x3.select_dtypes(include=["int64", "float64"]).columns
cat_cols3 = x3.select_dtypes(include=["string","category" , "object"]).columns
num_cols3 = num_cols3.copy()
preprocess3 = ColumnTransformer([
        ("num", "passthrough", num_cols3), 
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols3),
    ]
)
pipe3 = Pipeline([
    ("prep", preprocess3),
    ("rf", RFC (random_state=9, n_jobs=6, class_weight="balanced_subsample")) 
])
parmameter_test_3 = parmameter_test_1 
grid_personal3 = GSC(
    estimator = pipe3,
    cv=3,
    param_grid=parmameter_test_3,
    scoring='f1',
    n_jobs=6 #6 cores
)
grid_personal3.fit(x_train3, y_train3)
#test 3
best_personal3 = grid_personal3.best_estimator_
predict3_test = best_personal3.predict_proba(x_test3) [:, 1]
predict3_test_TH = (predict3_test >= 0.45).astype(int) #threshold
print(classification_report(y_test3, predict3_test_TH))
print(confusion_matrix(y_test3, predict3_test_TH))
print(grid_personal3.best_params_)
#model got better with eco values => I will use this one in future

#strongest parameters
best_cols3_test = permutation_importance(
    best_personal3, x_test3, y_test3,
    n_repeats=10,
    random_state=9,
    scoring="f1"
)
best_cols3 = pd.DataFrame({ #new dataframe
    "feature": x_test3.columns, #cols from bank_personal
    "importance_mean": best_cols3_test.importances_mean,
    "importance_std": best_cols3_test.importances_std,
}).sort_values("importance_mean", ascending=False)
print(best_cols3)

#save model
import joblib
joblib.dump(best_personal3, "bank_predict_y.joblib")

#Next chapter = clustering for better targeting