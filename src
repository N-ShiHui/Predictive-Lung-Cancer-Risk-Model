import sqlite3 as sql
import pandas as pd
import numpy as np

#Read dataset
db=sql.connect("lung_cancer.db")
sql_table = """SELECT name FROM sqlite_master WHERE type='table';"""
cursor=db.cursor()
c = cursor.execute(sql_table)
print(c.fetchall())

#transform db files into dataframes
LC= """ SELECT * FROM lung_cancer"""
data = pd.read_sql(LC,db)
print("Lung Cancer Data")
print(data.head(5))
print("")

#Clean and filter data
print('')
print(data.describe(percentiles=None, include=None, exclude=None))
data=data.drop(["ID"], axis=1)
data=data.dropna()


#Data Transformation and feature creation
#For ease of model building process, all object type data will be converted to integers

data['Stop Smoking']=data['Stop Smoking'].replace(['Still Smoking','Not Applicable'],[2024,0])
data['Start Smoking']=data['Start Smoking'].replace('Not Applicable',0)
data['Frequency of Tiredness']=data['Frequency of Tiredness'].replace('None / Low','Low')
data[['Stop Smoking','Start Smoking']]=data[['Stop Smoking','Start Smoking']].astype(int)
data['Smoking Duration(Years)']=data['Stop Smoking']-data['Start Smoking']
data['Weight Difference']=data['Current Weight']-data['Last Weight']
data=data.replace(['Male','MALE','Female','FEMALE','Yes','No','Present','Not Present','Low','Medium','High','Left','Right','RightBoth'],
                  [1,1,0,0,1,0,1,0,1,2,3,1,0,0])
#Drop irrelevant synthetic features and check on consistency
data=data.drop(['Start Smoking','Stop Smoking','Current Weight','Last Weight'], axis=1)
print(data.info())
print(data.head(20))


#Create train and test datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

#Based on the EDA done, 'WiFi' from data1 and 'Onboard Service' from data2 will be set as target variables
#Data1: Create training and testing datasets
#Only patients whom have Lung Cancer Occurrence data=1 will be considered in this analysis
d = ['Age','Gender','COPD History','Genetic Markers','Air Pollution Exposure',
      'Taken Bronchodilators', 'Frequency of Tiredness', 'Dominant Hand',
      'Smoking Duration(Years)','Weight Difference']
x = data.loc[:,d]
y = data['Lung Cancer Occurrence']==1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123, shuffle=True)
print('')
print('Lung Cancer trained dataset :')
print(x_train.head())
print('')
print(y_train.head())
print('')
print('Lung Cancer test dataset :')
print(x_test.head())
print('')
print(y_test.head())
 

#Build ml models for Lung Cancer data

#Random Forest Regressor algorithm
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
yrf_pred = rf.predict(x_test)
#RF confusion matrix
rf_cm = confusion_matrix(y_train, yrf_pred)
#false positive and true positive rates
fpr_rf, tpr_rf, _ = roc_curve(y_test, yrf_pred)
#AUC score of RF model
auc = roc_auc_score(y_test, yrf_pred)
r = rf.feature_importances_

print('')
print('RF model confusion matrix')
print(rf_cm)
print('')
print('RF classification report')
print(classification_report(y_test,yrf_pred,zero_division=1))
print("RF correlation coefficients between Lung Cancer Occurrence and other features")
print(d,r)
print("RF AUC score:",auc)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

logreg = LogisticRegression(random_state=16, max_iter=1000)
logreg.fit(x_train, y_train)
ylr_pred = logreg.predict(x_test)
ylr_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, ylr_pred_proba)
auc = roc_auc_score(y_test, ylr_pred_proba)
lr_cm = metrics.confusion_matrix(y_test, ylr_pred)
coeff = logreg.coef_
lr_df=data[[d]]
lc=data['Lung Cancer Occurrence']
for lc, coef in zip(lr_df.columns, coeff):
    print(lc, d, coef)
rmse = np.sqrt(mean_squared_error(y_test, ylr_pred_proba))
r2 = r2_score(y_test, ylr_pred_proba)

print('Logit Confusion Matrix:')
print(lr_cm)
print('')
print('Logit Classification Report:')
print(classification_report(y_test, ylr_pred, zero_division=1))
print('RMSE: {:.2f}'.format(rmse))
print('R2 score: {:.5f}'.format(r2))

#Linear Regression
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)
yreg_pred_test = reg.predict(x_test) 
yreg_pred_train = reg.predict(x_train) 
yreg_cm = metrics.confusion_matrix(y_test, yreg_pred_test)
rmse_reg = np.sqrt(mean_squared_error(y_test, yreg_pred_test))
r2_reg = r2_score(y_test, yreg_pred_test)

print("Linear Regression classification report:")
print(classification_report(y_test, yreg_pred_test, zero_division=1))
print("Confusion matrix for logit model:", yreg_cm)
print(f'Coefficient: {reg.coef_}')
print(f'Intercept: {reg.intercept_}')
print('RMSE: {:.2f}'.format(rmse_reg))
print('R2 score: {:.5f}'.format(r2_reg))

