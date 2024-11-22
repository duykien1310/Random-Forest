import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing: Encoding categorical variables
categorical_columns = [
    'person_gender', 'person_education', 'person_home_ownership', 
    'loan_intent', 'previous_loan_defaults_on_file'
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Splitting features and target
X_train = train_data.drop(columns=['loan_status'])
y_train = train_data['loan_status']

X_test = test_data.drop(columns=['loan_status'])
y_test = test_data['loan_status']

# Fill missing values if any
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
print("\nF1 Score (Weighted):", f1)
