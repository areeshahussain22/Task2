import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load and prep data
data = pd.read_csv('customer_churn_data.csv')
data.drop("CustomerID", axis=1, inplace=True)

X = data.drop("Churn", axis=1)
y = data["Churn"].map({"Yes": 1, "No": 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up preprocessing
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Model pipelines
logreg_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier())
])

# Tune models
print("Tuning models...")
logreg_search = GridSearchCV(logreg_pipe, 
                           {"model__C": [0.01, 0.1, 1, 10], 
                            "model__solver": ["liblinear"]},
                           cv=5, n_jobs=-1).fit(X_train, y_train)

rf_search = GridSearchCV(rf_pipe,
                       {"model__n_estimators": [100, 200],
                        "model__max_depth": [5, 10, None],
                        "model__min_samples_split": [2, 5]},
                       cv=5, n_jobs=-1).fit(X_train, y_train)

print("Logistic Regression:")
print(classification_report(y_test, logreg_search.predict(X_test)))

print("Random Forest:")
print(classification_report(y_test, rf_search.predict(X_test)))

best_model = rf_search if rf_search.best_score_ > logreg_search.best_score_ else logreg_search
joblib.dump(best_model.best_estimator_, "churn_model.joblib")
