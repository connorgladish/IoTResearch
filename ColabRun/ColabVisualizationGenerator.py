# https://www.github.com/connorgladish/IoTResearch/
"""
Generate all visualizations from ACI IoT results JSON
Includes: Confusion Matrices, ROC Curves, Comparisons, Feature Importance
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ============================================
# LOAD JSON DATA
# ============================================
print("📂 Loading results from JSON...")

# Change this filename to match your JSON file (IMPORTANT AND WILL VARY IF YOU MOVED IT OR RENAMED IT. CHECK!)
JSON_FILENAME = 'aci_comprehensive_results_with_knn.json'

try:
    with open(JSON_FILENAME, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded: {JSON_FILENAME}\n")
except FileNotFoundError:
    print(f"❌ Error: '{JSON_FILENAME}' not found!")
    print("Please make sure the JSON file is in the same directory as this script.")
    exit()

# ============================================
# CONFIGURE MATPLOTLIB
# ============================================
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# ============================================
# 1. BINARY CONFUSION MATRICES (2x2 Grid)
# ============================================
print("1️⃣ Creating binary confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

binary_models = data['binary_classification']['all_models']
model_names = ['Decision Tree', 'Random Forest', 'XGBoost', 'KNN']
cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']

for idx, (model_name, cmap) in enumerate(zip(model_names, cmaps)):
    cm_data = binary_models[model_name]['confusion_matrix']
    cm = np.array([[cm_data['tn'], cm_data['fp']],
                   [cm_data['fn'], cm_data['tp']]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx],
                xticklabels=['Predicted\nBenign', 'Predicted\nAttack'],
                yticklabels=['Actual\nBenign', 'Actual\nAttack'],
                cbar=True, annot_kws={"size": 14, "weight": "bold"},
                linewidths=2, linecolor='black')
    
    # Add metrics
    acc = binary_models[model_name]['accuracy']
    f1 = binary_models[model_name]['f1_score']
    auc = binary_models[model_name]['auc_roc']
    axes[idx].set_title(f'{model_name}\nAcc: {acc*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc:.4f}',
                       fontsize=12, fontweight='bold')
    
plt.suptitle('Binary Classification Confusion Matrices (All Models)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('1_binary_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 1_binary_confusion_matrices.png")

# ============================================
# 2. ROC-AUC CURVES (All Models - Full View)
# ============================================
print("2️⃣ Creating ROC-AUC curves (full view)...")

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
linestyles = ['-', '-', '-', '-']
roc_curves = data['binary_classification']['roc_curves']

for (model_name, color, ls) in zip(model_names, colors, linestyles):
    roc_data = roc_curves[model_name]
    fpr = np.array(roc_data['fpr'])
    tpr = np.array(roc_data['tpr'])
    auc_score = binary_models[model_name]['auc_roc']
    
    ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2.5, 
            label=f'{model_name} (AUC = {auc_score:.4f})')

# Plot random classifier line
ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier (AUC = 0.5000)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold')
ax.set_title('ROC Curves - Binary Classification (All Models)', 
             fontsize=16, fontweight='bold')
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('2_roc_curves_full.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 2_roc_curves_full.png")

# ============================================
# 3. ROC-AUC CURVES (Zoomed - Top-Left Corner)
# ============================================
print("3️⃣ Creating ROC-AUC curves (zoomed view)...")

fig, ax = plt.subplots(figsize=(10, 8))

for (model_name, color, ls) in zip(model_names, colors, linestyles):
    roc_data = roc_curves[model_name]
    fpr = np.array(roc_data['fpr'])
    tpr = np.array(roc_data['tpr'])
    auc_score = binary_models[model_name]['auc_roc']
    
    ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2.5, 
            label=f'{model_name} (AUC = {auc_score:.4f})')

# Zoom to top-left corner
ax.set_xlim([0.0, 0.1])
ax.set_ylim([0.9, 1.0])
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold')
ax.set_title('ROC Curves - Zoomed View (Top-Left Corner)', 
             fontsize=16, fontweight='bold')
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('3_roc_curves_zoomed.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 3_roc_curves_zoomed.png")

# ============================================
# 4. MULTI-CLASS CONFUSION MATRIX
# ============================================
print("4️⃣ Creating multi-class confusion matrix...")

cm_multi = np.array(data['multiclass_classification']['confusion_matrix'])
class_names = data['dataset_info']['class_names']

fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Number of Samples'},
            linewidths=0.5, linecolor='gray',
            annot_kws={"size": 9})

# Highlight diagonal (correct predictions)
for i in range(len(class_names)):
    ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))

best_multi = data['multiclass_classification']['best_model']
acc_multi = data['multiclass_classification']['all_models'][best_multi]['accuracy']

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Label', fontsize=14, fontweight='bold')
ax.set_title(f'Multi-Class Confusion Matrix - {best_multi}\n'
             f'Accuracy: {acc_multi*100:.2f}%',
             fontsize=16, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('4_multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 4_multiclass_confusion_matrix.png")

# ============================================
# 5. BINARY MODEL COMPARISON TABLE
# ============================================
print("5️⃣ Creating binary model comparison table...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

# Prepare data
table_data = []
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Time (min)']

for model_name in model_names:
    model_data = binary_models[model_name]
    table_data.append([
        model_name,
        f"{model_data['accuracy']*100:.2f}%",
        f"{model_data['precision']*100:.2f}%",
        f"{model_data['recall']*100:.2f}%",
        f"{model_data['f1_score']*100:.2f}%",
        f"{model_data['auc_roc']:.4f}",
        f"{model_data['time_min']:.2f}"
    ])

table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model row
best_binary = data['binary_classification']['best_model']
if best_binary in model_names:
    best_idx = model_names.index(best_binary) + 1
    for i in range(len(headers)):
        table[(best_idx, i)].set_facecolor('#FFF9C4')

plt.title('Binary Classification Model Comparison', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig('5_binary_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 5_binary_model_comparison.png")

# ============================================
# 6. MULTI-CLASS MODEL COMPARISON TABLE
# ============================================
print("6️⃣ Creating multi-class model comparison table...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

multiclass_models = data['multiclass_classification']['all_models']
table_data = []
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (min)']

for model_name in model_names:
    model_data = multiclass_models[model_name]
    table_data.append([
        model_name,
        f"{model_data['accuracy']*100:.2f}%",
        f"{model_data['precision']*100:.2f}%",
        f"{model_data['recall']*100:.2f}%",
        f"{model_data['f1_score']*100:.2f}%",
        f"{model_data['time_min']:.2f}"
    ])

table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#FF5722')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model row
best_multi = data['multiclass_classification']['best_model']
if best_multi in model_names:
    best_multi_idx = model_names.index(best_multi) + 1
    for i in range(len(headers)):
        table[(best_multi_idx, i)].set_facecolor('#FFF9C4')

plt.title('Multi-Class Classification Model Comparison', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig('6_multiclass_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 6_multiclass_model_comparison.png")

# ============================================
# 7. PER-CLASS PERFORMANCE BAR CHART
# ============================================
print("7️⃣ Creating per-class performance chart...")

per_class = data['multiclass_classification']['per_class_metrics']

# Extract data (exclude summary rows)
classes = [k for k in per_class.keys() 
           if k not in ['accuracy', 'macro avg', 'weighted avg']]
precision = [per_class[c]['precision'] for c in classes]
recall = [per_class[c]['recall'] for c in classes]
f1 = [per_class[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3')
bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800')

ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Per-Class Performance Metrics - {best_multi}', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars (only for bars with height < 0.95 to avoid clutter)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height < 0.95 and height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('7_per_class_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 7_per_class_performance.png")

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================
print("8️⃣ Creating feature importance chart...")

features = [f['feature'] for f in data['top_features']]
importances = [f['importance'] for f in data['top_features']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(features, importances, color='#9C27B0')
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Feature Importances (Multi-Class Classification)', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importances)):
    ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('8_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 8_feature_importance.png")

# ============================================
# 9. AUC SCORE COMPARISON (Bar Chart)
# ============================================
print("9️⃣ Creating AUC score comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

aucs = [binary_models[m]['auc_roc'] for m in model_names]
bars = ax.bar(model_names, aucs, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('AUC-ROC Score Comparison (Binary Classification)', 
             fontsize=14, fontweight='bold')
ax.set_ylim([0.99, 1.001])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, auc in zip(bars, aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.0005,
            f'{auc:.6f}',
            ha='center', va='top', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('9_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: 9_auc_comparison.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*60)
print("\n📁 Generated files:")
print("  1. 1_binary_confusion_matrices.png (2x2 grid)")
print("  2. 2_roc_curves_full.png (complete ROC curves)")
print("  3. 3_roc_curves_zoomed.png (zoomed top-left corner)")
print("  4. 4_multiclass_confusion_matrix.png (11x11 matrix)")
print("  5. 5_binary_model_comparison.png (comparison table)")
print("  6. 6_multiclass_model_comparison.png (comparison table)")
print("  7. 7_per_class_performance.png (bar chart)")
print("  8. 8_feature_importance.png (top 10 features)")
print("  9. 9_auc_comparison.png (AUC bar chart)")
print("\n✅ All images are complete!")
