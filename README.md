# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. start

STEP 2. Import pandas

STEP 3. Import Decision tree classifier

STEP 4. Fit the data in the model

STEP 5. Find the accuracy score

STEP 6. end

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NARRA RAMYA
RegisterNumber:  212223040128
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![decision tree classifier model](sam.png)
## Data.head()
![image](https://github.com/user-attachments/assets/342b1da2-f08f-4ea6-862d-f0f83a61c393)
## Data.info()
![image](https://github.com/user-attachments/assets/b7fb59a8-72b2-4db3-9e16-d9a83f94cf47)
![image](https://github.com/user-attachments/assets/c07e26dd-c7eb-4a8d-ae89-1af9db440584)
![image](https://github.com/user-attachments/assets/e8c07629-7010-43e4-87da-0de117a82ec7)
![image](https://github.com/user-attachments/assets/46d45d1f-a819-4c2a-91e0-33eeb78bc07d)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
