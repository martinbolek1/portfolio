import pandas as pd
import sqlalchemy as sqla
import numpy as np

bank = pd.read_csv("bank-additional-full.csv", sep=";")
bank_backup = bank.copy()
print(bank.columns.tolist()) #cols names

#File corruption
rng = np.random.default_rng(9)  # 9 is seed
cols_corrupt = ["age", "loan", "campaign"]
corruption_rate = 0.10  # 10% values in selected cols will be corrupted 
masks = {} #we need to save mask for future validation
for col in cols_corrupt:
    mask = rng.random(len(bank)) < corruption_rate
    masks[col] = mask #save mask
    bank.loc[mask, col] = np.nan
# corruption check
print(bank[cols_corrupt].isna().mean())

#File recovery - numeric values via KNN, loan variable will be predicted via logistic regression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
#Scale numeric features - KNN relies on distances, so scaling is important to prevent large-scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(bank[["age", "campaign"]])
#n_neighbors: number of nearest neighbors, weights="distance" = closer neighbors have higher influence
imputer = KNNImputer(n_neighbors=7, weights="distance")
imputed_scaled = imputer.fit_transform(scaled)
imputed_values = scaler.inverse_transform(imputed_scaled) #back from scaled values
#Put the imputed numeric data back into the dataframe
bank[["age", "campaign"]] = imputed_values
#Check if values are not missing
print(bank[cols_corrupt].isna().sum().sort_values(ascending=False).head(10))

#Validation of imputation just for fun :D
from sklearn.metrics import mean_absolute_error, mean_squared_error
for col in bank[["age", "campaign"]]:
    idx = masks[col]  # positions we removed in this column
    y_true = bank_backup.loc[idx, col]
    y_pred = bank.loc[idx, col]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{col}: MAE={mae:.3f}, RMSE={rmse:.3f}")
#Results are good

#Now let´s recover loan values via logistic regression
#preprocessing
loan_missing = bank["loan"].isna() | (bank["loan"] == "unknown") #there will be "1" where value is missing
loan_missing.sum() #number of rows without loan value
bank_with_loan = bank.loc[~loan_missing].copy() #train
bank_without_loan = bank.loc[loan_missing].copy() #test
y_loan = bank_with_loan["loan"].map({"no": 0, "yes": 1}) #train - y as category
#I need to do it this way to separate category and numeric variables in future
drop_cols = [c for c in ["loan", "y"] if c in bank.columns] #y is out aswell due to possible leakage
x_known = bank_with_loan.drop(columns=drop_cols) #train
x_miss  = bank_without_loan.drop(columns=drop_cols) #test
#same cols for x_miss as x_known => don't need to to this below for x_miss again
num_cols = x_known.select_dtypes(include=["int64", "float64"]).columns
cat_cols = x_known.select_dtypes(include=["object"]).columns

#now let's scale numeric variables and transform category variables to numeric
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),#handle_unknown="ignore" if there is unknown category in test set, it won't crash

    ]
)
x_loan_train = preprocess.fit_transform(x_known)
x_loan_test = preprocess.transform(x_miss)

#logistic regression
from sklearn.linear_model import LogisticRegression
loan_model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga")
loan_model.fit(x_loan_train, y_loan)
loan_recovered = loan_model.predict_proba(x_loan_test)[:, 1] #[:, 1] - I can just take one number(0/1), in this case 1 
#return values back to dataset
bank.loc[loan_missing, "loan"] = np.where(loan_recovered >= 0.57, "yes", "no") #0.57 best I tried 
#validation of logistic regression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_true = bank_backup.loc[idx, "loan"]   #real values
y_pred = bank.loc[idx, "loan"]          #predicted values
valid = y_true.isin(["yes", "no"])  #remove NaN(unknown)
y_true = y_true[valid]
y_pred = y_pred[valid]
print(confusion_matrix(y_true, y_pred, labels=["no","yes"]))
print(classification_report(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))

""" Best result accuracy: 0.98 %
        precision    recall  f1-score   support
no        0.98        1.00      0.99      3432
yes       0.97        0.91      0.94      645
"""

#Clearing Data
bank = bank.drop_duplicates()
bank_report = bank.copy()
bank_ml = bank.copy()
#Bank Report cleanup
obj_cols = bank_report.select_dtypes(include="object").columns #selecting objects 
bank_report[obj_cols] = bank_report[obj_cols].replace(r'^\s*$', np.nan, regex=True) #all NaN converted together
bank_report["new_customer"] = (bank_report["pdays"] != 999).astype(int) #new customer (999+ days old last contact)
bank_report.to_csv("bank_report.csv", index=False)
#Bank Science cleanup
obj_cols = bank_ml.select_dtypes(include="object").columns #selecting objects 
bank_ml[obj_cols] = bank_ml[obj_cols].replace(r'^\s*$', np.nan, regex=True) 
bank_ml[obj_cols] = bank_ml[obj_cols].fillna("unknown")
bank_ml["y"] = bank_ml["y"].map({"no":0, "yes":1})
bank_ml = bank_ml[bank_ml["y"].isin([0, 1])].copy() #delete rows without y=0/1
bank_ml.to_csv("bank_ml.csv", index=False)

#Upload to Database
import sqlalchemy as sqla
engine=sqla.create_engine("postgresql://postgres:1234@localhost:5432/bank_marketing")
bank_ml.to_sql(
    "bank_ml",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)
bank_report.to_sql(
    "bank_report",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)
#Next chapter: predictions for y variable - has the client subscribed a term deposit?