# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare the Dataset
Load Placement_Data.csv into a DataFrame.

Make a copy of the original data to work on (data1).

Drop unnecessary columns: sl_no (just an index) and salary (not known before placement).

2. Handle Missing and Duplicate Data
Check for missing values with isnull().sum().

Check for duplicate rows with duplicated().sum().

3. Encode Categorical Columns
Use LabelEncoder to convert categorical features into numeric values.

The following columns are encoded:

gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation, and status.

4. Split Features and Target
x = all columns except status (input features).

y = status column (target variable: 1 = placed, 0 = not placed).

5. Train-Test Split
Use train_test_split() to divide data:

80% training

20% testing

Set random_state=0 for reproducibility

6. Train Logistic Regression Model
Use LogisticRegression with liblinear solver.

Fit the model on training data (x_train, y_train).

7. Predict and Evaluate
Predict placement status for x_test.

Use:

accuracy_score to compute accuracy

confusion_matrix to see TP, FP, FN, TN

classification_report for precision, recall, F1-score

8. Make a Custom Prediction

Predict placement status for a new student record:

[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]
These values represent:

Encoded gender, ssc %, board, hsc %, board, stream, degree %, type, workex, etc.
## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ROHITH V
RegisterNumber: 212224220083  

```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
Placement data

![Screenshot 2025-03-28 105026](https://github.com/user-attachments/assets/91d05db0-2837-4f06-b508-2c420035c9ec)


Salary data

![Screenshot 2025-03-28 105042](https://github.com/user-attachments/assets/267bf203-c04f-4852-9047-6e8c803ba451)


Checking null function


![Screenshot 2025-03-28 105053](https://github.com/user-attachments/assets/97b43ff9-86fd-4723-8064-4d43936499df)


Data duplicate

![Screenshot 2025-03-28 105100](https://github.com/user-attachments/assets/f0723727-dc4b-4711-b682-143317b7efd6)


Print data

![Screenshot 2025-03-28 105123](https://github.com/user-attachments/assets/e6611498-58ed-4277-bb37-c7ae71157501)


Data status

![Screenshot 2025-03-28 105203](https://github.com/user-attachments/assets/58b58826-d88a-4e2a-8a8f-a1988d10b3ca)


Y-prediction array

![Screenshot 2025-03-28 105213](https://github.com/user-attachments/assets/f959e448-a08a-4c09-9479-5d4fe8d758fd)


Accuracy value

![Screenshot 2025-03-28 105443](https://github.com/user-attachments/assets/0bc94271-543a-4713-92db-32d9e81ae95d)


Confusion array

![Screenshot 2025-03-28 105225](https://github.com/user-attachments/assets/ec942e5b-fab8-49fe-9c63-8d6c5bc28e97)


Classification report

![Screenshot 2025-03-28 105241](https://github.com/user-attachments/assets/33f46544-b752-4e84-95dc-051cc2646385)


Prediction of LR

![Screenshot 2025-03-28 105254](https://github.com/user-attachments/assets/9c90dca1-1394-4673-922a-be25d982f95d)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
