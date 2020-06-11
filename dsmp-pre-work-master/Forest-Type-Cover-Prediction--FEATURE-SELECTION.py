# --------------
import pandas as pd
from sklearn import preprocessing
#path : File path
# Code starts here
# read the dataset
dataset=pd.read_csv(path)
# look at the first five columns
print(dataset.head())
# Check if there's any column which is not useful and remove it like the column id
dataset=dataset.drop(['Id'],axis=1)
# check the statistical description



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt
#names of all the attributes 
cols=dataset.columns.values
#number of attributes (exclude target)
size=len(cols)-1
#x-axis has target attribute to distinguish between classes
x=dataset['Cover_Type']
#y-axis shows values of an attribute
y=dataset.drop('Cover_Type',1)
# sns.violinplot(x=x,y=y,data=dataset) 
#Plot violin for all attributes


# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5
# Code Starts Here
subset_train=dataset.iloc[:,0:10]
data_corr=subset_train.corr()
sns.heatmap(data_corr)
correlation=data_corr.unstack().sort_values(kind='quicksort')
corr_var_list=correlation[((correlation>upper_threshold) | (correlation<lower_threshold))]
corr_var_list=corr_var_list[corr_var_list!=1]
print(corr_var_list)
# Code ends here


# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
Y=dataset['Cover_Type']
X=dataset.drop('Cover_Type',1)
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
#Standardized
#Apply transform only for continuous data
# size=len(X_train)-1
# print(size)
scaler=StandardScaler()
X_train_temp=scaler.fit_transform(X_train.iloc[:,0:size])
X_test_temp=scaler.fit_transform(X_test.iloc[:,0:size])

# X_train_temp=pd.DataFrame(X_train_temp)
# X_test_temp=pd.DataFrame(X_test_temp)

#Concatenate scaled continuous data and categorical

# print(X_train_temp.shape)
# print(X_train.shape)
# print(X_train[:,-1])
X_train1=numpy.concatenate((X_train_temp,X_train.iloc[:,size:]),axis=1)
X_test1=numpy.concatenate((X_test_temp,X_test.iloc[:,size:]),axis=1)

scaled_features_train_df=pd.DataFrame(data=X_train1, columns=X_train.columns, index=X_train.index)
scaled_features_test_df=pd.DataFrame(data=X_test1, columns=X_test.columns, index=X_test.index)

# print(scaled_features_train_df.head())



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
import numpy as np
# Write your solution here:

skb=SelectPercentile(score_func=f_classif, percentile=90)
predictors=skb.fit_transform(X_train1,Y_train)
scores=list(skb.scores_)
Features=scaled_features_train_df.columns
# print(predictors.shape[1])
# print(Features)
# Features.remove("Cover_Type")
# df1=pd.DataFrame(Features)
# df2=pd.DataFrame(scores)
# dataframe=pd.concat([df1,df2], axis=1)
# dataframe.columns=['Features', 'scores'] 
# dataframe.sort_values(by='scores', ascending=False)
dataframe=pd.DataFrame({'Features':Features,'Scores':scores}).sort_values(by='Scores', ascending=False)
print(dataframe.head())
# dataframe=pd.DataFrame(dataframe)
top_k_predictors=list(dataframe['Features'][:predictors.shape[1]])
# top_k_predictors=dataframe.Features.percentile(0.9)
# top_k_predictors=np.percentile(Features, 90)
print(top_k_predictors)




# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf=OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())

model_fit_all_features=clf1.fit(X_train,Y_train)
predictions_all_features=model_fit_all_features.predict(X_test)
score_all_features=accuracy_score(Y_test,predictions_all_features)
print(score_all_features)

model_fit_top_features=clf.fit(scaled_features_train_df[top_k_predictors],Y_train)
predictions_top_features=model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
score_top_features=accuracy_score(Y_test,predictions_top_features)
print(score_top_features)



