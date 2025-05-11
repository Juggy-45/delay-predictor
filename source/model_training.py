import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# Load the data set

df = pd.read_csv("../data/ireland_projects.csv")

# Encode categorical features

label_encoders = {}
categorical_cols = ['Project_Type', 'County']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and label
X = df.drop(columns=['Project_ID', 'Delay'])
y = df['Delay']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and encoders
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/delay_model.pkl")
joblib.dump(label_encoders, "../model/label_encoders.pkl")
print("Model and encoders saved.")