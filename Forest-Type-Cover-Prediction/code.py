# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset= pd.read_csv(path)
print(dataset.head())

dataset.drop(columns=['Id'],axis=1,inplace=True)

print(dataset.describe())

# look at the first five columns


# Check if there's any column which is not useful and remove it like the column id


# check the statistical description



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 

cols= dataset.columns
print(cols)


#number of attributes (exclude target)
size= len(dataset)
print(size)

#x-axis has target attribute to distinguish between classes
x=dataset['Cover_Type'].copy()


#y-axis shows values of an attribute
y=dataset.drop(['Cover_Type'],axis=1)

#Plot violin for all attributes
plt.violinplot(x)




# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here

subset_train=dataset.iloc[:,0:10]
data_corr= subset_train.corr(method='pearson')

sns.heatmap(data_corr)

correlation= data_corr.unstack().sort_values(kind='quicksort')

corr_var_list=correlation[(abs(correlation)>upper_threshold) & (correlation != 1)]
print(corr_var_list)


# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import numpy as np

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)




# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.

X= dataset.drop(['Cover_Type'],axis=1)
Y= dataset['Cover_Type'].copy()

X_train,X_test,Y_train,Y_test= cross_validation.train_test_split(X,Y,test_size=0.2,random_state=0)



#Standardized
#Apply transform only for continuous data

scaler= StandardScaler()

X_train_temp=X_train.iloc[:,0:10]
X_test_temp=X_test.iloc[:,0:10]

X_train_temp= scaler.fit_transform(X_train_temp)
X_test_temp= scaler.fit_transform(X_test_temp)

#Concatenate scaled continuous data and categorical


X_train1=np.concatenate((X_train_temp,X_train.iloc[:,10:]),axis=1)
X_test1=np.concatenate((X_test_temp,X_test.iloc[:,10:]),axis=1)
X_train_col= X_train.columns
features=[]
for i in X_train_col:
    features.append(i)


scaled_features_train_df =pd.DataFrame(X_train1,columns=features,index=X_train.index)
scaled_features_test_df = pd.DataFrame(X_test1,columns=features,index=X_test.index)

print(scaled_features_train_df)


"""X_temp = StandardScaler().fit_transform(X_train[:,0:size])
X_val_temp = StandardScaler().fit_transform(X_val[:,0:size])
#Concatenate non-categorical data and categorical
X_con = numpy.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con = numpy.concatenate((X_val_temp,X_val[:,size:]),axis=1)
#Add this version of X to the list 
X_all.append(['StdSca','All', X_con,X_val_con,1.0,cols,rem,ranks,i_cols,i_rem])"""







# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


skb=SelectPercentile(score_func=f_classif, percentile=20)

predictors= skb.fit(X_train1,Y_train)

scores= predictors.scores_


Features= X_train.columns
dataframe= pd.DataFrame({'Features':Features, 'scores':scores})
dat=dataframe.sort_values(by=['scores'],ascending=False)
print(dat)
top_k_predictors= list(dat['Features'].head(11))

print(top_k_predictors)



# Write your solution here:




# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

model=LogisticRegression()

clf=OneVsRestClassifier(estimator=model)
clf1=OneVsRestClassifier(estimator=model)

model_fit_all_features= clf1.fit(X_train,Y_train)
predictions_all_features=model_fit_all_features.predict(X_test)
score_all_features= accuracy_score(Y_test,predictions_all_features)
print(score_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors],Y_train)
predictions_top_features=model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
score_top_features= accuracy_score(Y_test,predictions_top_features)
print(score_top_features)




