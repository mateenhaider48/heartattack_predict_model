import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder    
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

df = pd.read_csv('heart.csv')
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True) 
numeric_cols =  ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
       'Oldpeak', 'HeartDisease', 'ChestPainType_ATA', 'ChestPainType_NAP',
       'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST',
       'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
df[numeric_cols] = df[numeric_cols].astype(int)



X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

models = {
    "logistic Regression":LogisticRegression(),
    "KNN":KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":GaussianNB(),
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "SVM":SVC(kernel='rbf', random_state=42)
}
 
results = [] 

for name,model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, f1))

print("Model Performance:")
for name, acc, f1 in results:
    print(f"{name}: Accuracy = {acc * 100:.4f} , F1 Score = {f1 * 100:.4f}")

import joblib
joblib.dump(models['SVM'], 'heart_disease_model.pkl')
joblib.dump(scalar, 'scaler.pkl')
joblib.dump(X.columns, 'feature_columns.pkl')    