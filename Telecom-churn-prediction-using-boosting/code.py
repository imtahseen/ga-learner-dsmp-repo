# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here

df=pd.read_csv(path)

X=df.drop(columns=['customerID','Churn'],axis=1)

y=df['Churn'].copy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)





# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
       
X_train['TotalCharges']=X_train['TotalCharges'].apply(lambda x:str(x).strip()).replace('', np.nan)
X_test['TotalCharges']=X_test['TotalCharges'].apply(lambda x:str(x).strip()).replace('', np.nan)

X_train['TotalCharges']=X_train['TotalCharges'].astype('float')
X_test['TotalCharges']=X_test['TotalCharges'].astype('float')

X_train['TotalCharges']=X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean())
X_test['TotalCharges']=X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean())

print(X_train.isnull().sum())

en=LabelEncoder()

catx=X_train.select_dtypes(include='object')
for i in catx:
    X_train[i]=en.fit_transform(X_train[i])
for j in catx:
    X_test[j]=en.fit_transform(X_test[j])

y_train=y_train.apply(lambda x:x.replace('No','0'))
y_train=y_train.apply(lambda x:x.replace('Yes','1'))
y_test=y_test.apply(lambda x:x.replace('No','0'))
y_test=y_test.apply(lambda x:x.replace('Yes','1'))

y_train=y_train.astype('int')
y_test=y_test.astype('int')



# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here

ada_model=AdaBoostClassifier(random_state=0)

ada_model.fit(X_train,y_train)

y_pred=ada_model.predict(X_test)

ada_score=accuracy_score(y_test,y_pred)
ada_cm=confusion_matrix(y_test,y_pred)
ada_cr=classification_report(y_test,y_pred)

print(ada_score)
print(ada_cm)
print(ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here

xgb_model=XGBClassifier(random_state=0)

xgb_model.fit(X_train,y_train)

y_pred= xgb_model.predict(X_test)

xgb_score= xgb_model.score(X_test,y_test)

xgb_cm=confusion_matrix(y_test,y_pred)

xgb_cr= classification_report(y_test,y_pred)

print(xgb_score)
print(xgb_cm)
print(xgb_cr)

xgb_clf =XGBClassifier(random_state=0)
clf_model=GridSearchCV(estimator= xgb_clf,param_grid=parameters)

clf_model.fit(X_train,y_train)

y_pred= clf_model.predict(X_test)

clf_score= accuracy_score(y_test,y_pred)

clf_cm=confusion_matrix(y_test,y_pred)

clf_cr= classification_report(y_test,y_pred)


print(clf_score)
print(clf_cm)
print(clf_cr)



