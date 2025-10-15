import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load and prepare data
data = pd.read_csv('dummy_adolescent_concern_dataset.csv')

# Encode categorical features
categorical_cols = ['Gender', 'Parental_Education', 'Ad_Exposure_Freq', 'Health_Knowledge', 'Parental_Influence', 'Concern_Level']
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop('Concern_Level', axis=1)
y = data['Concern_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define base models
rf = RandomForestClassifier(random_state=42, n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True, random_state=42)
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
lr = LogisticRegression(random_state=42, max_iter=1000)

# Stacking ensemble
stacking_clf = StackingClassifier(
    estimators=[('rf', rf), ('knn', knn), ('svm', svm), ('xgb', xgb)],
    final_estimator=lr,
    cv=5
)
stacking_clf.fit(X_train_resampled, y_train_resampled)

# Predictions and metrics
y_pred = stacking_clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'F1 Score: {f1_score(y_test, y_pred, average="weighted"):.4f}')

# Feature importances from Random Forest
rf.fit(X_train_resampled, y_train_resampled)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

# SHAP for local explainability (on the first test instance)
explainer = shap.Explainer(rf, X_train_resampled)
shap_values = explainer(X_test.iloc[0:1])
print("\nSHAP Explanation for First Test Instance (top 5 features per class):")
for i, class_name in enumerate(['High', 'Low', 'Medium']):
    top_features = shap_values.values[0, :, i].argsort()[-5:][::-1]
    print(f"Class {class_name}:")
    for idx in top_features:
        print(f"  {X.columns[idx]}: {shap_values.values[0, idx, i]:.4f}")

# Save the model
joblib.dump(stacking_clf, 'model.pkl')
print("Model saved as 'model.pkl'")