# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the spam dataset (spam.csv) using appropriate encoding and check for missing values.

2. Extract email text (v2) as input features and labels (v1 – spam/ham) as target output.

3.Divide the dataset into training and testing sets using train-test split method.

4.Apply Count Vectorizer to transform email text into numerical feature vectors.

5. Train the SVM classifier using training data, predict results on test data, and calculate accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SAI KRIPA SK 
RegisterNumber: 212224040284 
*/
# Detect file encoding
import chardet

with open('spam.csv', 'rb') as file:
    encoding = chardet.detect(file.read(100000))
print(encoding)


# Load dataset
import pandas as pd

data = pd.read_csv('spam.csv', encoding='Windows-1252')

print(data.head())
print(data.isnull().sum())


# Define features and labels
X = data['v2']   # email text
y = data['v1']   # spam / ham


# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# Convert text to numerical data
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Train model
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)

print("Predicted values:", y_pred)


# Accuracy
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Output:
<img width="929" height="45" alt="image" src="https://github.com/user-attachments/assets/529d05af-8d5f-4a5e-b742-732ad1360ecc" />
<img width="958" height="493" alt="image" src="https://github.com/user-attachments/assets/ee2d6b61-41b0-4724-9f75-89f21a5f98a3" />
<img width="704" height="63" alt="image" src="https://github.com/user-attachments/assets/95afe09a-45af-4217-baa2-097527c18562" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
