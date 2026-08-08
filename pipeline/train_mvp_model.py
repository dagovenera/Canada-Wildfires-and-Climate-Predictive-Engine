import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =============================================
# STEP 1: LOAD DATA & COMPUTE ROLLING FEATURES
# =============================================
print("🔄 Loading clean analysis-ready dataset...")
df = pd.read_csv("clean_analysis_ready_data.csv")

# Ensure time sorting to prevent issues during rolling calculations
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by=["Station Name", "Date"])

print("⚙️ Engineering rolling climate features (Drought Indicators)...")
df["Rolling_Mean_Temp_7d"] = df.groupby("Station Name")["Mean Temp (°C)"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

df["Rolling_Precip_7d"] = df.groupby("Station Name")["Total Precip (mm)"].transform(
    lambda x: x.rolling(7, min_periods=1).sum()
)

# Safely handle wind gust missing data
df["Spd of Max Gust (km/h)"] = df["Spd of Max Gust (km/h)"].fillna(0)


# ==========================================
# STEP 2: DEFINE FEATURES AND TARGET LABEL
# ==========================================
feature_cols = [
    "Mean Temp (°C)", 
    "Total Precip (mm)", 
    "Spd of Max Gust (km/h)", 
    "Rolling_Mean_Temp_7d", 
    "Rolling_Precip_7d"
]

X = df[feature_cols]
y = df["Is_Fire_Event"].astype(int)


# ==========================================
# STEP 3: STRATIFIED TRAIN-TEST SPLIT (THE FIX)
# ==========================================
print("✂️ Executing stratified split to preserve minority fire classes...")

# Using stratify=y guarantees both sets get a matching ratio of fire events
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


# ==========================================
# STEP 4: MODEL TRAINING (XGBOOST CLASSIFIER)
# ==========================================
print("🚀 Initializing and training XGBoost Model...")

num_negative_cases = (y_train == 0).sum()
num_positive_cases = (y_train == 1).sum()
imbalance_ratio = num_negative_cases / max(1, num_positive_cases)

# model initialization
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=imbalance_ratio,  # Balances target weight vectors
    max_delta_step=1,  #FIX CLASS STARVATION CONVERGENCE
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


# ==========================================
# STEP 5: RIGOROUS METRICS EVALUATION
# ==========================================
print("\n📊 --- MACHINE LEARNING EVALUATION METRICS --- 📊\n")
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Dynamic check: ensure metrics layer safely handles unexpected single-class conditions
unique_classes = np.unique(y_val)

if len(unique_classes) > 1:
    print("1. Classification Profile:")
    print(classification_report(y_val, y_pred, target_names=["No Fire", "Fire Event"]))
    
    print("2. Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"\n3. ROC-AUC Score: {auc_score:.4f}")
else:
    print("⚠️ Validation set contains only one class. Evaluation skipped.")
    print(f"Total entries evaluated: {len(y_val)} (All rows are class {unique_classes[0]})")


# ==========================================
# STEP 6: EXPORT PRODUCTION ARTIFACTS
# ==========================================
print("\n💾 Archiving trained weights for deployment...")
model_artifacts = {
    "model": model,
    "features": feature_cols
}
joblib.dump(model_artifacts, "canadian_wildfire_model.pkl")
print("🎉 Success! File exported as: canadian_wildfire_model.pkl")
