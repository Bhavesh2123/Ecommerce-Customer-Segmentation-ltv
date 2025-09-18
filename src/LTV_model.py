#Load & Feature Engineering
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance
import shap
import joblib

df = pd.read_csv("Cleaned_df.csv")

#extract features from a datetime column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceYear'] = df['InvoiceDate'].dt.year
df['InvoiceMonth'] = df['InvoiceDate'].dt.month
df['InvoiceDay'] = df['InvoiceDate'].dt.day
df['InvoiceWeekday'] = df['InvoiceDate'].dt.weekday

# Drop columns not usable by XGBoost
drop_cols = ['InvoiceDate','Description','StockCode','Country']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target variable
y = df['LTV']                 
X = df.drop(columns=['LTV'])

# Convert any remaining object columns to category for native handling
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train XGBoost Model

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True
)

model.fit(X_train, y_train)

#Evaluate

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")

# Optional cross-validation RMSE
cv_scores = -cross_val_score(model, X, y,
                             scoring='neg_root_mean_squared_error',
                             cv=5)
print(f"Cross-validated RMSE: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

#Feature Importance
fig, ax = plt.subplots(figsize=(8,6))
plot_importance(model, importance_type='gain', ax=ax)
plt.show()


#SHAP Explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)

#Save for Dashboard
joblib.dump(model, "xgb_ltv_model.pkl")
X.to_csv("X_features_for_dashboard.csv", index=False)
