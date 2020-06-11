# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here

df=pd.read_csv(path)

# print(df.head())

# print(df.info())

# cols=df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]

# df.INCOME.apply(lambda x: x.replace('$',''))
df["INCOME"] = [str(x).replace('$','') for x in df["INCOME"]]
df["HOME_VAL"] = [str(x).replace('$','') for x in df["HOME_VAL"]]
df["BLUEBOOK"] = [str(x).replace('$','') for x in df["BLUEBOOK"]]
df["OLDCLAIM"] = [str(x).replace('$','') for x in df["OLDCLAIM"]]
df["CLM_AMT"] = [str(x).replace('$','') for x in df["CLM_AMT"]]

df["INCOME"] = [str(x).replace(',','') for x in df["INCOME"]]
df["HOME_VAL"] = [str(x).replace(',','') for x in df["HOME_VAL"]]
df["BLUEBOOK"] = [str(x).replace(',','') for x in df["BLUEBOOK"]]
df["OLDCLAIM"] = [str(x).replace(',','') for x in df["OLDCLAIM"]]
df["CLM_AMT"] = [str(x).replace(',','') for x in df["CLM_AMT"]]


# cols.apply(lambda x: x.replace(',',''))

print(df.head())

X=df.drop("CLAIM_FLAG",1)
y=df.CLAIM_FLAG.copy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)

# Code ends here


# --------------
# Code starts here

X_train["INCOME"] = X_train.INCOME.astype(float)
X_train["HOME_VAL"] = X_train.HOME_VAL.astype(float)
X_train["BLUEBOOK"] = X_train.BLUEBOOK.astype(float)
X_train["OLDCLAIM"] = X_train.OLDCLAIM.astype(float)
X_train["OLDCLAIM"] = X_train.OLDCLAIM.astype(float)

X_test["INCOME"] = X_test.INCOME.astype(float)
X_test["HOME_VAL"] = X_test.HOME_VAL.astype(float)
X_test["BLUEBOOK"] = X_test.BLUEBOOK.astype(float)
X_test["OLDCLAIM"] = X_test.OLDCLAIM.astype(float)
X_test["OLDCLAIM"] = X_test.OLDCLAIM.astype(float)

# print(X_train.null().sum())

# print(X_test.null().sum())

# Code ends here


# --------------
# Code starts here

X_train=X_train.dropna(subset=['YOJ', 'OCCUPATION'])
X_test=X_test.dropna(subset=['YOJ', 'OCCUPATION'])

y_train=y_train[X_train.index]
y_test=y_test[X_test.index]

# columns=['AGE','CAR_AGE','INCOME','HOME_VAL']

# for col in columns:
#     X_train[col] = X_train[col].fillna((X_train[col].mean()))
#     X_test[col] = X_test[col].fillna((X_test[col].mean()))

X_train['AGE'] = X_train['AGE'].fillna((X_train['AGE'].mean()))
X_train['CAR_AGE'] = X_train['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()))
X_train['INCOME'] = X_train['INCOME'].fillna((X_train['INCOME'].mean()))
X_train['HOME_VAL'] = X_train['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()))


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here

for col in columns:
    le=LabelEncoder()
    # X_train[col]=X_train[col].astype(str)
    # X_test[col]=X_test[col].astype(str)
    X_train[col]=le.fit_transform(X_train[col].astype(str))
    X_test[col]=le.transform(X_test[col].astype(str))

# df['label'] = le.fit_transform(df['label'])

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

model=LogisticRegression(random_state = 6)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

score=model.score(X_test,y_test)
print(score)

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

#Initialising the model
smote = SMOTE(random_state=9)

#Undersampling the data using cluster centroids
X_train, y_train = smote.fit_sample(X_train, y_train)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here

#Initialising logistic regression model
model = LogisticRegression()

#Fitting the model with sampled data
model.fit(X_train, y_train)

#Making predictions on test data
y_pred=model.predict(X_test)

#Finding the accuracy score
score=accuracy_score(y_test,y_pred)
print("Accuracy:",score)  

# Code ends here


