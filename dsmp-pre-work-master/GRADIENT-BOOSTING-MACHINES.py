# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df=pd.read_csv(path)
X= df.drop(['customerID','Churn'],1)
y=df['Churn'].copy()
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)





# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train['TotalCharges']=X_train['TotalCharges'].replace(" ",np.NaN)
X_test['TotalCharges']=X_test['TotalCharges'].replace(" ",np.NaN)

X_train['TotalCharges']=X_train['TotalCharges'].astype(float)
X_test['TotalCharges']=X_test['TotalCharges'].astype(float)

X_train['TotalCharges'].fillna((X_train['TotalCharges'].mean()), inplace=True)
X_test['TotalCharges'].fillna((X_test['TotalCharges'].mean()), inplace=True)

# print(X_train.isnull().sum())

# cat=df.columns
# cat.remove("tenure","customerID","MonthlyCharges","TotalCharges")
a=list(X_train.select_dtypes(object).columns)
print(a)

# X_train.apply(LabelEncoder().fit_transform)
# X_test.apply(LabelEncoder().fit_transform)
# 
le=LabelEncoder()
X=X_train
for i in range(0,X.shape[1]):
    if X.dtypes[i]=='object':
        X[X.columns[i]] = le.fit_transform(X[X.columns[i]])

X1=X_test
for i in range(0,X1.shape[1]):
    if X1.dtypes[i]=='object':
        X1[X1.columns[i]] = le.fit_transform(X1[X1.columns[i]])

# le.fit_transform(X_train['gender'])
# le.fit_transform(X_test['gender'])

# le.fit_transform(X_train['Partner'])
# le.fit_transform(X_test['Partner'])

# le.fit_transform(X_train['Dependents'])
# le.fit_transform(X_test['Dependents'])

# le.fit_transform(X_train['PhoneService'])
# le.fit_transform(X_test['PhoneService'])

# le.fit_transform(X_train['MultipleLines'])
# le.fit_transform(X_test['MultipleLines'])

# le.fit_transform(X_train['InternetService'])
# le.fit_transform(X_test['InternetService'])

# le.fit_transform(X_train['OnlineSecurity'])
# le.fit_transform(X_test['OnlineSecurity'])

# le.fit_transform(X_train['OnlineBackup'])
# le.fit_transform(X_test['OnlineBackup'])

# le.fit_transform(X_train['DeviceProtection'])
# le.fit_transform(X_test['DeviceProtection'])

# le.fit_transform(X_train['TechSupport'])
# le.fit_transform(X_test['TechSupport'])

# le.fit_transform(X_train['StreamingTV'])
# le.fit_transform(X_test['StreamingTV'])

# le.fit_transform(X_train['StreamingMovies'])
# le.fit_transform(X_test['StreamingMovies'])


# le.fit_transform(X_train['Contract'])
# le.fit_transform(X_test['Contract'])

# le.fit_transform(X_train['PaperlessBilling'])
# le.fit_transform(X_test['PaperlessBilling'])

# le.fit_transform(X_train['PaymentMethod'])
# le.fit_transform(X_test['PaymentMethod'])

y_train=y_train.replace({'No':0, 'Yes':1})
y_test=y_test.replace({'No':0, 'Yes':1})





# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here

# Fitting with Adaboost
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
y_pred=ada_model.predict(X_test)
ada_score=accuracy_score(y_pred,y_test)

ada_cm=confusion_matrix(y_test,y_pred)
ada_cr=classification_report(y_test,y_pred)





# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here

xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

#Finding the accuracy score
xgb_score = accuracy_score(y_test,y_pred)
print("Accuracy: ",xgb_score)

#Finding the confusion matrix
xgb_cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n', xgb_cm)

#Finding the classification report
xgb_cr=classification_report(y_test,y_pred)
print('Classification report: \n', xgb_cr)


clf_model=GridSearchCV(estimator=xgb_model, param_grid=parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)


clf_score = accuracy_score(y_test,y_pred)
print("Accuracy: ",clf_score)

#Finding the confusion matrix
clf_cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n', clf_cm)

#Finding the classification report
clf_cr=classification_report(y_test,y_pred)
print('Classification report: \n', clf_cr)




