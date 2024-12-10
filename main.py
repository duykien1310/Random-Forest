import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Hàm tính Gini Index
def gini_index(groups, classes):
    n_instances = sum([len(group) for group in groups])  # Tổng số mẫu
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        class_counts = np.bincount(group, minlength=len(classes))
        for count in class_counts:
            p = count / size
            score += p * p
        gini += (1 - score) * (size / n_instances)
    return gini

# Hàm chia dữ liệu
def split_data(X, y, feature_idx, threshold):
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    return y[left_mask], y[right_mask]

# Hàm tìm split tốt nhất
def best_split(X, y, classes, n_features=None):
    n_samples, n_total_features = X.shape
    if n_features is None:
        n_features = n_total_features
    feature_indices = np.random.choice(n_total_features, n_features, replace=False)
    
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    best_groups = None

    for feature_idx in feature_indices:
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left, right = split_data(X, y, feature_idx, threshold)
            if len(left) == 0 or len(right) == 0:
                continue
            gini = gini_index([left, right], classes)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_idx
                best_threshold = threshold
                best_groups = (left, right)
    return best_feature, best_threshold, best_groups

# Hàm xây dựng cây quyết định
def build_tree(X, y, max_depth, min_samples_split, depth=0, n_features=None):
    classes = np.unique(y)
    if len(classes) == 1 or len(y) < min_samples_split or depth >= max_depth:
        return {'label': np.bincount(y).argmax()}
    
    feature, threshold, groups = best_split(X, y, classes, n_features)
    if feature is None or not groups or any(len(group) == 0 for group in groups):
        return {'label': np.bincount(y).argmax()}
    
    left, right = groups
    left_tree = build_tree(X[X[:, feature] <= threshold], left, max_depth, min_samples_split, depth + 1, n_features)
    right_tree = build_tree(X[X[:, feature] > threshold], right, max_depth, min_samples_split, depth + 1, n_features)
    
    return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

# Hàm dự đoán cho từng mẫu
def predict_single(tree, x):
    if 'label' in tree:
        return tree['label']
    if x[tree['feature']] <= tree['threshold']:
        return predict_single(tree['left'], x)
    else:
        return predict_single(tree['right'], x)

# Hàm Random Forest Training
def random_forest_train(X, y, n_trees, max_depth, min_samples_split, n_features):
    forest = []
    for _ in range(n_trees):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sample, y_sample = X[indices], y[indices]
        tree = build_tree(X_sample, y_sample, max_depth, min_samples_split, n_features=n_features)
        forest.append(tree)
    return forest

# Hàm dự đoán với rừng cây
def random_forest_predict(forest, X):
    predictions = []
    for x in X:
        tree_preds = [predict_single(tree, x) for tree in forest]
        predictions.append(np.bincount(tree_preds).argmax())
    return np.array(predictions)

# Đánh giá mô hình
def evaluate_model(y_true, y_pred):
    """
    Đánh giá mô hình dựa trên các độ đo:
    - Confusion Matrix
    - Precision, Recall, F1-score
    """
    print("### Đánh giá mô hình ###")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Tính Precision, Recall, và F1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\nPrecision (Weighted):", precision)
    print("Recall (Weighted):", recall)
    print("F1-Score (Weighted):", f1)
    
    # In chi tiết từng lớp trong Classification Report
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:\n", report)
    
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# Đọc dữ liệu
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Chọn các feature quan trọng
selected_features = [
    'previous_loan_defaults_on_file', 
    'loan_percent_income', 
    'loan_int_rate', 
    'person_income', 
    'person_home_ownership', 
    'loan_intent', 
    'credit_score'
]

# Tiền xử lý dữ liệu
train_data = train_data[selected_features + ['loan_status']]
test_data = test_data[selected_features + ['loan_status']]

# Mã hóa các cột phân loại
categorical_columns = ['person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

X_train = train_data.drop(columns=['loan_status']).values
y_train = train_data['loan_status'].values
X_test = test_data.drop(columns=['loan_status']).values
y_test = test_data['loan_status'].values

# Điền giá trị thiếu
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Huấn luyện mô hình
forest = random_forest_train(X_train, y_train, n_trees=100, max_depth=15, min_samples_split=30, n_features=3)

# Dự đoán
y_pred = random_forest_predict(forest, X_test)

# Đánh giá model
evaluation_results = evaluate_model(y_test, y_pred)
