import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Any, Tuple


# Custom Label Encoder
class CustomLabelEncoder:
    def __init__(self):
        self.classes_ = {}

    def fit(self, data: pd.Series):
        unique_classes = sorted(data.unique())
        self.classes_ = {cls: idx for idx, cls in enumerate(unique_classes)}
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        return data.map(self.classes_)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)


# Decision Tree Implementation
class DecisionTree:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = X.copy()
        data['target'] = y
        self.tree = self._build_tree(data)

    def _build_tree(self, data: pd.DataFrame, depth: int = 0) -> Any:
        if depth == self.max_depth or len(data['target'].unique()) == 1:
            return self._majority_class(data['target'])

        best_split = self._find_best_split(data)
        if not best_split:
            return self._majority_class(data['target'])

        left, right = best_split['left'], best_split['right']
        feature, value = best_split['feature'], best_split['value']

        return {
            'feature': feature,
            'value': value,
            'left': self._build_tree(left, depth + 1),
            'right': self._build_tree(right, depth + 1)
        }

    def _find_best_split(self, data: pd.DataFrame) -> dict:
        best_gini = float('inf')
        best_split = None

        for feature in data.columns[:-1]:
            unique_values = data[feature].unique()
            for value in unique_values:
                left = data[data[feature] <= value]
                right = data[data[feature] > value]

                if len(left) == 0 or len(right) == 0:
                    continue

                gini = self._gini_impurity(left['target'], right['target'])
                if gini < best_gini:
                    best_gini = gini
                    best_split = {'feature': feature, 'value': value, 'left': left, 'right': right}

        return best_split

    def _gini_impurity(self, left: pd.Series, right: pd.Series) -> float:
        def gini(group: pd.Series) -> float:
            probs = group.value_counts(normalize=True)
            return 1.0 - sum(probs**2)

        total = len(left) + len(right)
        return (len(left) / total) * gini(left) + (len(right) / total) * gini(right)

    def _majority_class(self, y: pd.Series) -> int:
        return y.mode()[0]

    def predict_row(self, row: pd.Series) -> int:
        node = self.tree
        while isinstance(node, dict):
            if row[node['feature']] <= node['value']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, X: pd.DataFrame) -> List[int]:
        return [self.predict_row(row) for _, row in X.iterrows()]


# Random Forest Implementation
class RandomForest:
    def __init__(self, n_estimators: int, max_depth: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for _ in range(self.n_estimators):
            bootstrap_sample = self._bootstrap(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(*bootstrap_sample)
            self.trees.append(tree)

    def _bootstrap(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        indices = np.random.choice(X.index, size=len(X), replace=True)
        return X.loc[indices], y.loc[indices]

    def predict(self, X: pd.DataFrame) -> List[int]:
        predictions = [tree.predict(X) for tree in self.trees]
        majority_votes = [Counter(tree_preds).most_common(1)[0][0] for tree_preds in zip(*predictions)]
        return majority_votes


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
    le = CustomLabelEncoder()
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

# Train the Random Forest model
rf_model = RandomForest(n_estimators=10, max_depth=5)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")

# Display predictions
test_data['Prediction'] = y_pred
test_data['Prediction_Label'] = test_data['Prediction'].apply(lambda x: 'Cho vay' if x == 1 else 'Không cho vay')

print("\nKết quả dự đoán trên file test.csv:")
print(test_data[['person_age', 'loan_status', 'Prediction_Label']])
