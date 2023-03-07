import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option("display.max_colwidth", 1000)

LCdata_dict = pd.read_excel("LCDataDictionary.xlsx", sheet_name="LoanStats", skiprows = [i for i in range(79, 100) ])
LCdata_dict.index = np.arange(1, len(LCdata_dict) + 1)
LCdata_dict[["Features", "Description"]]


df = pd.read_csv('loan_data_2007_2014.csv')
df = df.drop('Unnamed: 0', axis=1)
df.sample(5)


print("Total Rows :", df.shape[0])
print("Total Features :", df.shape[1])
print("Duplicate Value:", df.duplicated().sum())
list_df = []
for col in df.columns:
    list_df.append([col, df[col].dtype, df[col].isna().sum(), 100*df[col].isna().sum()/len(df[col])])
df_desc = pd.DataFrame(data=list_df, columns=['Feature','Data Type','Null', 'Null (%)'])
df_desc


drop_list = df_desc[df_desc['Null (%)'] > 25]['Feature'] #drop feature that have more than 25% missing value
drop_list = list(drop_list) + ['member_id', 'id'] #drop member_id and id, because we don't need it

df.drop(labels = drop_list, axis = 1, inplace = True)

loan_status = df['loan_status'].value_counts()

plt.figure(figsize = (10, 5))
ax = sns.barplot(x = loan_status.values, y = loan_status.index)

for val in ax.containers:
    ax.bar_label(val,)

for stat in loan_status.index:
    
    print(stat, ':', f"{round((loan_status[stat] / 466285)  * 100, 2)}%")

good = ['Fully Paid'] 
bad = ['Charged Off', 'Does not meet the credit policy. Status:Fully Paid', 
       'Default', 'Does not meet the credit policy. Status:Charged Off']

def add_Label(values):  
    if values in good:
        return 1
    return 0

new_df = df[df['loan_status'].isin(good + bad)].copy()
new_df['loan_status'] = new_df['loan_status'].apply(add_Label)

new_df.sample(5)

status_percentage = 100*new_df['loan_status'].value_counts()/len(new_df['loan_status'])
status_percentage = pd.DataFrame(status_percentage)
status_percentage.rename(columns={"loan_status": "Status Percentage"}, inplace =True)
status_percentage.insert(0, "Status", ['Good','Bad'], True)

status_percentage

correlations = (new_df.select_dtypes(exclude=object)
                .corr()
                .dropna(how="all", axis=0)
                .dropna(how="all", axis=1)
)

correlations['loan_status'].sort_values(ascending = False)

have_cor = correlations[(correlations >= 0.5) & (correlations <= 0.9)]
plt.figure(figsize = (15, 10))
cor_plot = sns.heatmap(
                have_cor, 
                annot=True, 
                fmt=".2f",
                cmap="coolwarm",
                mask=np.triu(np.ones_like(have_cor, dtype=bool))
)

X = new_df.drop(columns="loan_status")
y = new_df.loan_status


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

numerical_feature = have_cor.columns[have_cor.notnull().any()].tolist()
numerical_feature
categorical_feature = ["grade", "emp_length", "home_ownership", "verification_status", "purpose"]
categorical_feature

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ("numeric", numerical_pipeline, numerical_feature),
    ("categoric", categorical_pipeline, categorical_feature)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("algo", LogisticRegression())
])

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)

pipeline.fit(X_train, y_train)
y_pred_proba = pipeline.predict_proba(X_test)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
AUC = roc_auc_score(y_test, y_pred_proba[:, 1])


print("Accuracy ",accuracy)
print("Recall ",recall)
print("Precision ",precision)
print("AUC ",AUC)

report = classification_report(y_true = y_train, y_pred = pipeline.predict(X_train))
print(report)

confusion = confusion_matrix(y_true = y_test, y_pred = pipeline.predict(X_test))
plt.figure(figsize=(5, 4))
sns.heatmap(confusion, annot=True, fmt="g")
plt.show()