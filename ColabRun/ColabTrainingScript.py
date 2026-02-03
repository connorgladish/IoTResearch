# https://github.com/connorgladish/IoTResearch/
"""
Train models on ACI IoT data - MULTI-CLASS with Decision Tree, KNN + ROC Data
"""

import pandas as pd
import numpy as np
import glob
import os

# ============================================
# LOAD DATA (AUTO-DETECT)
# ============================================
print("📊 Loading ACI IoT dataset...")

# List all CSV files in the main directory
print("\nAvailable files:")
!ls -lh /content/aci_iot_data/*.csv

# Find the main dataset file (not the Payload one)
csv_files = glob.glob('/content/aci_iot_data/*.csv')
csv_files = [f for f in csv_files if 'Payload' not in f and os.path.getsize(f) > 100*1024*1024]

if not csv_files:
    print("❌ No suitable CSV files found!")
    raise Exception("Dataset not found")

# Use the largest one
dataset_file = max(csv_files, key=lambda x: os.path.getsize(x))

file_size_gb = os.path.getsize(dataset_file) / (1024**3)
print(f"\n✅ Using: {os.path.basename(dataset_file)} ({file_size_gb:.1f} GB)")
print("⚠️ Loading - may take 5-15 minutes...\n")

try:
    df = pd.read_csv(dataset_file, low_memory=False)
    print(f"✅ Loaded {len(df):,} samples")
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative encoding...")
    df = pd.read_csv(dataset_file, low_memory=False, encoding='latin-1')
    print(f"✅ Loaded {len(df):,} samples")

print(f"✅ Shape: {df.shape}")
print(f"\n📋 Columns ({len(df.columns)}):")
print(df.columns.tolist())
print(f"\n🔍 First few rows:")
print(df.head())

# ============================================
# FIND LABEL
# ============================================
print("\n🎯 Finding label column...")

label_candidates = [
    'label', 'Label', 'LABEL',
    'attack', 'Attack', 'attack_type', 'Attack_Type',
    'class', 'Class', 'classification',
    'category', 'Category',
    'target', 'Target'
]

label_col = None

for col in label_candidates:
    if col in df.columns:
        label_col = col
        break

if not label_col:
    for col in df.columns:
        if any(kw in col.lower() for kw in ['label', 'attack', 'class', 'target', 'category']):
            label_col = col
            print(f"🔍 Found potential label: '{label_col}'")
            break

if not label_col:
    print("\n❌ Could not find label column.")
    print("\nAll columns:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] {col} - {df[col].nunique()} unique values")
    raise Exception("Please identify label column")

print(f"✅ Label column: '{label_col}'")
print(f"\n📊 Label Distribution:")
print(df[label_col].value_counts())

# ============================================
# FILTER OUT RARE CLASSES
# ============================================
print("\n🔍 Filtering rare attack classes...")

MIN_SAMPLES_PER_CLASS = 100  # Minimum samples to keep a class

class_counts = df[label_col].value_counts()
rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()

if rare_classes:
    print(f"  Removing {len(rare_classes)} rare classes with <{MIN_SAMPLES_PER_CLASS} samples:")
    for cls in rare_classes:
        print(f"    - {cls}: {class_counts[cls]} samples")

    df = df[~df[label_col].isin(rare_classes)].reset_index(drop=True)
    print(f"\n  Remaining samples: {len(df):,}")
    print(f"  Remaining classes: {df[label_col].nunique()}")

# ============================================
# PREPROCESS
# ============================================
print("\n🔄 Preprocessing...")

cols_to_drop = [label_col]
drop_keywords = ['id', 'timestamp', 'time', 'date', 'flow_id', 'src_ip', 'dst_ip']
for col in df.columns:
    if any(kw in col.lower() for kw in drop_keywords) and col != label_col:
        cols_to_drop.append(col)

X = df.drop(columns=list(set(cols_to_drop)), errors='ignore')
y_multiclass = df[label_col].copy()

print(f"  Features: {X.shape[1]}")

# ============================================
# MULTI-CLASS LABEL ENCODING
# ============================================
from sklearn.preprocessing import LabelEncoder

print(f"\n🏷️ Encoding multi-class labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_multiclass)

# Store class names
class_names = label_encoder.classes_
num_classes = len(class_names)

print(f"  Number of classes: {num_classes}")
print(f"  Classes: {class_names.tolist()}")

# Show distribution
unique, counts = np.unique(y_encoded, return_counts=True)
for cls, count in zip(class_names, counts):
    print(f"    {cls}: {count:,}")

# ============================================
# ALSO CREATE BINARY LABELS (for comparison)
# ============================================
normal_keywords = ['normal', 'benign', 'legitimate']
normal_value = None

for val in y_multiclass.unique():
    if any(kw in str(val).lower() for kw in normal_keywords):
        normal_value = val
        break

if normal_value:
    y_binary = (y_multiclass != normal_value).astype(int)
    print(f"\n  Binary - Normal: '{normal_value}'")
else:
    most_common = y_multiclass.value_counts().index[0]
    y_binary = (y_multiclass != most_common).astype(int)
    print(f"\n  Binary - Normal (assumed): '{most_common}'")

print(f"  Binary - Benign: {(y_binary==0).sum():,}")
print(f"  Binary - Attack: {(y_binary==1).sum():,}")

# ============================================
# ENCODE CATEGORICAL FEATURES
# ============================================
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f"\n  Encoding {len(cat_cols)} categorical feature columns...")
    for col in cat_cols:
        try:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        except:
            X = X.drop(columns=[col])

# Clean
X = X.fillna(X.mean(numeric_only=True))
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.select_dtypes(include=[np.number])

print(f"✅ Final: {X.shape[0]:,} samples × {X.shape[1]} features")

# ============================================
# STRATIFIED SAMPLING (CRITICAL FIX)
# ============================================
print("\n🎲 Stratified Sampling...")

from sklearn.model_selection import train_test_split

SAMPLE_SIZE = 500000

if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
    print(f"  Target: {SAMPLE_SIZE:,} samples from {len(X):,} total")

    # Use stratified sampling to preserve class distribution
    try:
        X_sampled, _, y_encoded_sampled, _, y_binary_sampled, _ = train_test_split(
            X, y_encoded, y_binary,
            train_size=SAMPLE_SIZE,
            random_state=42,
            stratify=y_encoded
        )

        X = X_sampled.reset_index(drop=True)
        y_encoded = y_encoded_sampled
        y_binary = y_binary_sampled.reset_index(drop=True)

        print(f"  ✅ Stratified sample: {len(X):,} samples")

        # Verify class distribution
        unique_sampled, counts_sampled = np.unique(y_encoded, return_counts=True)
        print(f"\n  Sampled class distribution:")
        for cls, count in zip(class_names, counts_sampled):
            print(f"    {cls}: {count:,}")

    except ValueError as e:
        print(f"  ⚠️ Stratified sampling failed: {e}")
        print(f"  Using random sampling instead...")
        idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y_encoded = y_encoded[idx]
        y_binary = y_binary.iloc[idx].reset_index(drop=True)

# ============================================
# SPLIT
# ============================================
print("\n✂️ Splitting into train/val/test...")

from sklearn.preprocessing import StandardScaler

# Check if any class has too few samples
min_samples = np.min(np.bincount(y_encoded))
print(f"  Minimum samples per class: {min_samples}")

if min_samples < 6:
    print(f"  ⚠️ Some classes have very few samples. Adjusting split strategy...")
    test_size = min(0.15, max(6 / len(X), 0.05))
else:
    test_size = 0.3

# Split for multi-class
X_train, X_temp, y_train_multi, y_temp_multi = train_test_split(
    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val_multi, y_test_multi = train_test_split(
    X_temp, y_temp_multi, test_size=0.5, random_state=42, stratify=y_temp_multi
)

# Split for binary (using same indices for consistency)
train_idx = X_train.index
temp_idx = X_temp.index
val_idx = X_val.index
test_idx = X_test.index

y_train_bin = y_binary.iloc[train_idx]
y_val_bin = y_binary.iloc[val_idx]
y_test_bin = y_binary.iloc[test_idx]

# Reset indices
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train_bin = y_train_bin.reset_index(drop=True)
y_val_bin = y_val_bin.reset_index(drop=True)
y_test_bin = y_test_bin.reset_index(drop=True)

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✅ Ready!\n")

# ============================================
# TRAIN - BINARY CLASSIFICATION
# ============================================
print("="*60)
print("🚀 BINARY CLASSIFICATION TRAINING")
print("="*60)

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import xgboost as xgb

binary_results = {}
roc_data_binary = {}  # NEW: Store ROC curve data

print("\n1️⃣ Decision Tree (Binary)...")
t0 = time.time()
dt_bin = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_bin.fit(X_train_scaled, y_train_bin)
dt_time = time.time() - t0

dt_pred = dt_bin.predict(X_test_scaled)
dt_proba = dt_bin.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test_bin, dt_proba)
roc_data_binary['Decision Tree'] = {
    'fpr': fpr_dt.tolist(),
    'tpr': tpr_dt.tolist(),
    'thresholds': thresholds_dt.tolist()
}

binary_results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test_bin, dt_pred),
    'precision': precision_score(y_test_bin, dt_pred, zero_division=0),
    'recall': recall_score(y_test_bin, dt_pred, zero_division=0),
    'f1': f1_score(y_test_bin, dt_pred, zero_division=0),
    'auc_roc': roc_auc_score(y_test_bin, dt_proba),
    'time_min': dt_time/60,
    'predictions': dt_pred
}

print(f"   ✅ Acc: {binary_results['Decision Tree']['accuracy']*100:.2f}%\n")

print("2️⃣ Random Forest (Binary)...")
t0 = time.time()
rf_bin = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, verbose=0)
rf_bin.fit(X_train_scaled, y_train_bin)
rf_time = time.time() - t0

rf_pred = rf_bin.predict(X_test_scaled)
rf_proba = rf_bin.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_bin, rf_proba)
roc_data_binary['Random Forest'] = {
    'fpr': fpr_rf.tolist(),
    'tpr': tpr_rf.tolist(),
    'thresholds': thresholds_rf.tolist()
}

binary_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test_bin, rf_pred),
    'precision': precision_score(y_test_bin, rf_pred, zero_division=0),
    'recall': recall_score(y_test_bin, rf_pred, zero_division=0),
    'f1': f1_score(y_test_bin, rf_pred, zero_division=0),
    'auc_roc': roc_auc_score(y_test_bin, rf_proba),
    'time_min': rf_time/60,
    'predictions': rf_pred
}

print(f"   ✅ Acc: {binary_results['Random Forest']['accuracy']*100:.2f}%\n")

print("3️⃣ XGBoost (Binary)...")
t0 = time.time()
xgb_bin = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
xgb_bin.fit(X_train_scaled, y_train_bin, verbose=False)
xgb_time = time.time() - t0

xgb_pred = xgb_bin.predict(X_test_scaled)
xgb_proba = xgb_bin.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test_bin, xgb_proba)
roc_data_binary['XGBoost'] = {
    'fpr': fpr_xgb.tolist(),
    'tpr': tpr_xgb.tolist(),
    'thresholds': thresholds_xgb.tolist()
}

binary_results['XGBoost'] = {
    'accuracy': accuracy_score(y_test_bin, xgb_pred),
    'precision': precision_score(y_test_bin, xgb_pred, zero_division=0),
    'recall': recall_score(y_test_bin, xgb_pred, zero_division=0),
    'f1': f1_score(y_test_bin, xgb_pred, zero_division=0),
    'auc_roc': roc_auc_score(y_test_bin, xgb_proba),
    'time_min': xgb_time/60,
    'predictions': xgb_pred
}

print(f"   ✅ Acc: {binary_results['XGBoost']['accuracy']*100:.2f}%\n")

print("4️⃣ KNN (Binary)...")
t0 = time.time()
knn_bin = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_bin.fit(X_train_scaled, y_train_bin)
knn_time = time.time() - t0

knn_pred = knn_bin.predict(X_test_scaled)
knn_proba = knn_bin.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test_bin, knn_proba)
roc_data_binary['KNN'] = {
    'fpr': fpr_knn.tolist(),
    'tpr': tpr_knn.tolist(),
    'thresholds': thresholds_knn.tolist()
}

binary_results['KNN'] = {
    'accuracy': accuracy_score(y_test_bin, knn_pred),
    'precision': precision_score(y_test_bin, knn_pred, zero_division=0),
    'recall': recall_score(y_test_bin, knn_pred, zero_division=0),
    'f1': f1_score(y_test_bin, knn_pred, zero_division=0),
    'auc_roc': roc_auc_score(y_test_bin, knn_proba),
    'time_min': knn_time/60,
    'predictions': knn_pred
}

print(f"   ✅ Acc: {binary_results['KNN']['accuracy']*100:.2f}%\n")

# ============================================
# TRAIN - MULTI-CLASS CLASSIFICATION
# ============================================
print("="*60)
print("🚀 MULTI-CLASS CLASSIFICATION TRAINING")
print("="*60)

multiclass_results = {}

print("\n1️⃣ Decision Tree (Multi-Class)...")
t0 = time.time()
dt_multi = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_multi.fit(X_train_scaled, y_train_multi)
dt_multi_time = time.time() - t0

dt_multi_pred = dt_multi.predict(X_test_scaled)

multiclass_results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test_multi, dt_multi_pred),
    'precision': precision_score(y_test_multi, dt_multi_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test_multi, dt_multi_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test_multi, dt_multi_pred, average='weighted', zero_division=0),
    'time_min': dt_multi_time/60,
    'predictions': dt_multi_pred
}

print(f"   ✅ Acc: {multiclass_results['Decision Tree']['accuracy']*100:.2f}%\n")

print("2️⃣ Random Forest (Multi-Class)...")
t0 = time.time()
rf_multi = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, verbose=0)
rf_multi.fit(X_train_scaled, y_train_multi)
rf_multi_time = time.time() - t0

rf_multi_pred = rf_multi.predict(X_test_scaled)

multiclass_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test_multi, rf_multi_pred),
    'precision': precision_score(y_test_multi, rf_multi_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test_multi, rf_multi_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test_multi, rf_multi_pred, average='weighted', zero_division=0),
    'time_min': rf_multi_time/60,
    'predictions': rf_multi_pred
}

print(f"   ✅ Acc: {multiclass_results['Random Forest']['accuracy']*100:.2f}%\n")

print("3️⃣ XGBoost (Multi-Class)...")
t0 = time.time()
xgb_multi = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss'
)
xgb_multi.fit(X_train_scaled, y_train_multi, verbose=False)
xgb_multi_time = time.time() - t0

xgb_multi_pred = xgb_multi.predict(X_test_scaled)

multiclass_results['XGBoost'] = {
    'accuracy': accuracy_score(y_test_multi, xgb_multi_pred),
    'precision': precision_score(y_test_multi, xgb_multi_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test_multi, xgb_multi_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test_multi, xgb_multi_pred, average='weighted', zero_division=0),
    'time_min': xgb_multi_time/60,
    'predictions': xgb_multi_pred
}

print(f"   ✅ Acc: {multiclass_results['XGBoost']['accuracy']*100:.2f}%\n")

print("4️⃣ KNN (Multi-Class)...")
t0 = time.time()
knn_multi = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_multi.fit(X_train_scaled, y_train_multi)
knn_multi_time = time.time() - t0

knn_multi_pred = knn_multi.predict(X_test_scaled)

multiclass_results['KNN'] = {
    'accuracy': accuracy_score(y_test_multi, knn_multi_pred),
    'precision': precision_score(y_test_multi, knn_multi_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test_multi, knn_multi_pred, average='weighted', zero_division=0),
    'f1': f1_score(y_test_multi, knn_multi_pred, average='weighted', zero_division=0),
    'time_min': knn_multi_time/60,
    'predictions': knn_multi_pred
}

print(f"   ✅ Acc: {multiclass_results['KNN']['accuracy']*100:.2f}%\n")

# ============================================
# BINARY RESULTS
# ============================================
print("="*60)
print("📊 BINARY CLASSIFICATION RESULTS")
print("="*60)

binary_results_df = pd.DataFrame({k: {m: v for m, v in v.items() if m != 'predictions'}
                                   for k, v in binary_results.items()}).T
print("\n", binary_results_df.to_string())

best_binary = max(binary_results.keys(), key=lambda k: binary_results[k]['f1'])
print(f"\n🏆 Best Binary Model: {best_binary}")

# Binary confusion matrices for all models
print(f"\n📊 Binary Confusion Matrices:\n")
for model_name, results in binary_results.items():
    cm = confusion_matrix(y_test_bin, results['predictions'])
    tn, fp, fn, tp = cm.ravel()
    print(f"{model_name}:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")
    print(f"  FPR: {fp/(fp+tn)*100:.2f}%  Detection: {tp/(tp+fn)*100:.2f}%\n")

# ============================================
# MULTI-CLASS RESULTS
# ============================================
print("="*60)
print("📊 MULTI-CLASS CLASSIFICATION RESULTS")
print("="*60)

multiclass_results_df = pd.DataFrame({k: {m: v for m, v in v.items() if m != 'predictions'}
                                       for k, v in multiclass_results.items()}).T
print("\n", multiclass_results_df.to_string())

best_multiclass = max(multiclass_results.keys(), key=lambda k: multiclass_results[k]['f1'])
print(f"\n🏆 Best Multi-Class Model: {best_multiclass}")

# Multi-class confusion matrix for best model
print(f"\n📊 Multi-Class Confusion Matrix ({best_multiclass}):\n")
cm_multi = confusion_matrix(y_test_multi, multiclass_results[best_multiclass]['predictions'])

print("Confusion Matrix Shape:", cm_multi.shape)
print("\nFull Matrix:")
print(cm_multi)

# Per-class metrics
print(f"\n📊 Per-Class Performance ({best_multiclass}):\n")
per_class_report = classification_report(
    y_test_multi,
    multiclass_results[best_multiclass]['predictions'],
    target_names=class_names,
    digits=4,
    zero_division=0
)
print(per_class_report)

# ============================================
# FEATURE IMPORTANCE
# ============================================
print(f"\n🔑 Top 10 Features ({best_multiclass} Multi-Class):")
best_multi_model = xgb_multi if best_multiclass == 'XGBoost' else (rf_multi if best_multiclass == 'Random Forest' else (dt_multi if best_multiclass == 'Decision Tree' else None))

if best_multi_model and hasattr(best_multi_model, 'feature_importances_'):
    importances = best_multi_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    print(feature_importance.to_string(index=False))
else:
    print("  ⚠️ KNN doesn't have feature importance")
    # Use Random Forest as fallback
    importances = rf_multi.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    print("  Using Random Forest feature importance instead:")
    print(feature_importance.to_string(index=False))

# ============================================
# SAVE COMPREHENSIVE RESULTS
# ============================================
import json

# Binary confusion matrix for best binary model
cm_bin = confusion_matrix(y_test_bin, binary_results[best_binary]['predictions'])
tn, fp, fn, tp = cm_bin.ravel()

paper_results = {
    'dataset_info': {
        'dataset_name': 'ACI-IoT-2023',
        'file': os.path.basename(dataset_file),
        'total_samples': len(df),
        'used_samples': len(X),
        'features': X.shape[1],
        'num_classes': int(num_classes),
        'class_names': class_names.tolist(),
        'class_distribution': {str(cls): int(cnt) for cls, cnt in zip(class_names, counts)}
    },
    'binary_classification': {
        'best_model': best_binary,
        'all_models': {
            model_name: {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1']),
                'auc_roc': float(results['auc_roc']),
                'time_min': float(results['time_min']),
                'confusion_matrix': {
                    'tn': int(confusion_matrix(y_test_bin, results['predictions']).ravel()[0]),
                    'fp': int(confusion_matrix(y_test_bin, results['predictions']).ravel()[1]),
                    'fn': int(confusion_matrix(y_test_bin, results['predictions']).ravel()[2]),
                    'tp': int(confusion_matrix(y_test_bin, results['predictions']).ravel()[3])
                }
            }
            for model_name, results in binary_results.items()
        },
        'roc_curves': roc_data_binary  # NEW: ROC curve data
    },
    'multiclass_classification': {
        'best_model': best_multiclass,
        'all_models': {
            model_name: {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1']),
                'time_min': float(results['time_min'])
            }
            for model_name, results in multiclass_results.items()
        },
        'confusion_matrix': cm_multi.tolist(),
        'per_class_metrics': classification_report(
            y_test_multi,
            multiclass_results[best_multiclass]['predictions'],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    },
    'top_features': feature_importance.to_dict('records')
}

print("\n📋 COMPREHENSIVE RESULTS SAVED\n")

with open('aci_comprehensive_results_with_knn.json', 'w') as f:
    json.dump(paper_results, f, indent=2)

from google.colab import files
files.download('aci_comprehensive_results_with_knn.json')

print("\n🎉 DONE!")
print("\n✅ You now have:")
print("  • 4 binary models (Decision Tree, Random Forest, XGBoost, KNN)")
print("  • 4 multi-class models (Decision Tree, Random Forest, XGBoost, KNN)")
print("  • Confusion matrices for all models")
print("  • Per-class performance metrics")
print("  • Feature importance rankings")
print("  • ROC curve data for all binary models")
