# STEP 1:Import libraries
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import Decision TreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklaern.metrics import (
f1_score,
accuracy_score,
    classification_report,
confusion_matrix
)
# STEP 2: Load dataset
data = load_brest_cancer()
X = pd.DataFrame(
  data.data,
  columns=data.feature_names
)
y = pd.Series(data.target,name="target")
print("Dataset shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget values:")
print(y.vaklue_counts())
# STEP 3: Split dataset
X_train,X_test,y_train,y_test = train_test_split(
  X,
  y,
  test_size=0.20,
  random_state=42,
  stratify=y
)
print("\nTraining samples:",X_train.shape[0])
print("Testing samples:",X_test.shape[0])
