# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages.
2.Print the present data and placement data and salary data.
3.Using logistic regression find the predicted values of accuracy confusion matrices.
4.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SONU S 
RegisterNumber: 212223220107
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data = data.drop(["sl_no", "salary"], axis = 1)
data.info()
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data
x = data.iloc[:, :-1].values
x
y=data.iloc[:, -1].values
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_predict,y_test))
print(confusion_matrix(y_predict,y_test))
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
clf.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]) 
*/
```

## Output:
PLACEMENT DATA:
![image](https://github.com/user-attachments/assets/eca5fdb9-3bdf-4d8c-8e98-0293fb0b03c6)

![image](https://github.com/user-attachments/assets/31cd158d-415b-42f0-bab2-025e93ed0dd0)

![image](https://github.com/user-attachments/assets/e2969a1e-2e74-4c2d-ab4e-21a3f33595d7)

![image](https://github.com/user-attachments/assets/a8ab4374-c10b-4d67-a306-ed40cea29dba)

Y_PREDICTED ARRAY:

![image](https://github.com/user-attachments/assets/321074fa-2d8c-4aa0-a55c-6b6c1f8158e7)

ACCURACY:

![image](https://github.com/user-attachments/assets/092e2a58-bf82-453e-84b3-a5c247b574a8)

CONFUSION MATRIX:

![image](https://github.com/user-attachments/assets/c2abd3e3-67b3-412f-a716-2bcc1893cf06)


![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
