# Machine Learning-Based Anomaly Detection for Traffic in IoT-Enabled Transportation Networks
[![Conference](https://img.shields.io/badge/Conference-CYBER--CARE%20Symposium-blue)](https://erau.edu)
[![Dataset](https://img.shields.io/badge/Dataset-ACI--IoT--2023-green)](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Machine Learning-Based Anomaly Detection for Traffic in IoT-Enabled Transportation Networks**
>
> Research to be presented in tandem with Molly Corgan at the **CYBER-CARE Symposium** at Embry-Riddle Aeronautical University

---

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Project Structure](#project-structure)

---
![V2V (Vehicle-To-Vehicle)](Figures/v2v.png)
> **Figure:** An illustration of Vehicle-to-Vehicle (V2V) communication in a connected transportation environment. Vehicles exchange real-time network traffic data with one another and with roadside infrastructure. This communication layer is the primary attack surface this research aims to protect — malicious traffic injected into these channels can compromise navigation, safety systems, and fleet management.

## Overview

This research evaluates **five machine learning algorithms** for detecting cyberattacks in IoT-enabled transportation networks. As connected vehicles, V2V communication, and smart infrastructure become critical to modern transportation, protecting these systems from cyber threats is paramount.

We trained and compared:
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **1D Convolutional Neural Network (CNN)**

Using the **ACI-IoT-2023 dataset** (1.2M+ samples, 11 attack types), we achieved **>99.5% accuracy** across all classical models, with XGBoost demonstrating optimal performance for real-time deployment. The CNN was added as a deep learning baseline, achieving 99.24% binary and 98.34% multi-class accuracy.

---

## Research Motivation

### Why Transportation Cybersecurity Matters

Connected transportation systems face increasing cyber threats:

- **Vehicle-to-Vehicle (V2V) Communication**: Vulnerable to man-in-the-middle attacks
- **Smart Traffic Infrastructure**: Target for DDoS attacks disrupting traffic flow
- **Autonomous Vehicles**: Susceptible to sensor spoofing and network intrusion
- **Fleet Management Systems**: Risk of data exfiltration and ransomware

### Real-World Attack Example: Adversarial Perception Manipulation

![Adversarial Attack on Autonomous Vehicle](Figures/attackexample.png)
> **Figure:** A side-by-side comparison showing the effect of an adversarial network attack on an autonomous vehicle's semantic segmentation model. The top panel shows normal operation — the AI correctly identifies road surfaces, pedestrians, vehicles, and surrounding objects. The bottom panel shows the same scene after an attacker injects adversarial noise through a compromised V2X channel: the AI misclassifies a pedestrian as drivable road surface, meaning the vehicle would accelerate toward them. This example motivates the need for real-time network intrusion detection before such payloads reach the perception pipeline.

Demonstration of semantic segmentation attack on autonomous vehicle perception.

The diagram above illustrates a critical vulnerability in autonomous driving systems if the attack is not detected and suppressed:

1. **Normal Operation (Top)**: The vehicle's camera captures an urban scene and correctly identifies the scene through semantic segmentation. The AI classifies road surface (purple), buildings/non important features (grey), trees (yellow), vehicles (blue), sidewalks (pink), and pedestrians (red), enabling safe navigation decisions.

2. **Under Attack (Bottom)**: An attacker compromises the vehicle's network and injects adversarial noise into the perception pipeline. While the scene appears identical to human observers, the AI's segmentation model **fails catastrophically**—the pedestrian is misclassified as drivable road surface (purple). The vehicle would now accelerate directly toward the pedestrian, believing the path is clear.

**This is not theoretical**: Research has demonstrated that adversarial attacks can be executed through:
- Compromised V2X communication channels
- Malicious Over-The-Air (OTA) updates
- Physical adversarial patches on road signs
- GPS spoofing combined with camera manipulation

**Our Goal**: Develop lightweight, accurate ML models capable of **real-time threat detection** in resource-constrained IoT environments.

---

## Dataset

### ACI-IoT-2023 Dataset

- **Total Samples**: 1,231,406 network flows
- **Features**: 78 numerical/categorical features
- **Attack Types**: 11 distinct classes
- **Source**: [Army Cyber Institute (ACI)](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023)

#### Class Distribution

| Attack Type | Samples | Percentage |
|-------------|---------|------------|
| Port Scan | 441,282 | 35.8% |
| Benign | 329,295 | 26.7% |
| ICMP Flood | 225,234 | 18.3% |
| Ping Sweep | 71,928 | 5.8% |
| DNS Flood | 46,935 | 3.8% |
| Vulnerability Scan | 39,537 | 3.2% |
| OS Scan | 37,524 | 3.0% |
| Slowloris | 18,643 | 1.5% |
| SYN Flood | 13,857 | 1.1% |
| Dictionary Attack | 6,380 | 0.5% |
| UDP Flood | 791 | 0.06% |

**Note**: Rare classes (<100 samples) like ARP Spoofing were filtered to ensure reliable model training.

---

## Training Pipeline

All experiments were implemented in Python 3.8 using scikit-learn, XGBoost, TensorFlow/Keras, and NumPy, and executed on Google Colab with GPU acceleration. The pipeline proceeded through the following steps.

### Step 1 — Data Ingestion and Rare Class Removal

The raw ACI-IoT-2023 dataset was loaded from a single CSV file (~88.8 GB) containing 1,231,411 network flow records and 78 features. Attack classes with fewer than 100 samples were removed entirely, as they cannot be reliably partitioned into stratified train/validation/test subsets. ARP Spoofing fell below this threshold and was excluded. The filtered dataset retained **11 attack classes plus Benign**.

### Step 2 — Feature Matrix Construction and Target Label Encoding

The label column was isolated and two target label arrays were constructed in parallel:

- **Multi-class**: All class names encoded as integers (0–10) using `LabelEncoder`
- **Binary**: Records matching "benign", "normal", or "legitimate" assigned label `0`; all others assigned label `1`

Non-predictive identifier columns (IP addresses, timestamps, flow IDs) were dropped at this stage.

### Step 3 — Categorical Feature Encoding and Data Cleaning

All object/categorical columns were encoded to integers using `LabelEncoder` applied independently per column. Missing values (NaN) were imputed using the column-wise arithmetic mean. Infinite values — which arise in flow-derived statistics when denominators approach zero — were replaced with NaN and subsequently filled with zero. Final feature dimensionality: approximately **47–50 features**.

### Step 4 — Stratified Downsampling to 500,000 Records

Training KNN and Random Forest on 1M+ records is computationally prohibitive in a single-session environment. A stratified sample of exactly **500,000 records** was drawn using `train_test_split` with `stratify=y_encoded` and `random_state=42`. This ensures each class retains proportional representation. UDP Flood's 791 total records produced ~321 sampled records, which is the proximate cause of that class's degraded recall at test time.

### Step 5 — Train / Validation / Test Split (70% / 15% / 15%)

The 500,000-record sample was partitioned into three non-overlapping subsets using two sequential stratified splits:

- **Training set**: 70% (~350,000 records)
- **Validation set**: 15% (~75,000 records)
- **Test set**: 15% (~75,000 records)

The test set was isolated before any model fitting and accessed only once, at final evaluation.

### Step 6 — Feature Standardization

All features were standardized to zero mean and unit variance using `StandardScaler`. The scaler was **fit exclusively on the training set** and applied without refitting to the validation and test sets, preventing data leakage. Although Decision Tree and Random Forest are scale-invariant, XGBoost gradient updates, KNN distance computations, and CNN gradient flow are all sensitive to feature magnitude — uniform scaling ensures performance differences reflect algorithmic properties rather than scaling artifacts.

### Step 7 — Binary Classification Training

Five classifiers were trained on the binary-labeled training set:

| Model | Key Hyperparameters |
|-------|---------------------|
| Decision Tree | `max_depth=20` |
| Random Forest | `n_estimators=100`, `max_depth=20` |
| XGBoost | `n_estimators=100`, `max_depth=10`, `lr=0.1`, `objective=binary:logistic` |
| KNN | `k=5`, Euclidean distance |
| CNN | `Adam(lr=0.001)`, `batch_size=512`, `EarlyStopping(patience=4)`, ran 13 epochs |

Wall-clock training time was recorded for each model. After training, predicted class labels and predicted probabilities were generated on the held-out test set. ROC curve data were computed and stored for all five models.

### Step 8 — Multi-Class Classification Training

The same five classifiers were retrained on the multi-class-labeled training set with two modifications for XGBoost and CNN:

- **XGBoost**: objective changed to `multi:softmax`, `num_class=11`, `eval_metric=mlogloss`
- **CNN**: output layer changed from a single sigmoid unit to an 11-unit softmax layer; ran 9 epochs before early stopping

All other hyperparameters remained identical to ensure a controlled comparison between tasks. Weighted-average precision, recall, and F1-score were computed alongside per-class metrics for all 11 attack types.

### Step 9 — Evaluation and Result Serialization

For each model and each task, the following metrics were computed on the held-out test set: accuracy, weighted precision, weighted recall, weighted F1-score, AUC-ROC (binary only), confusion matrix (raw cell counts), per-class classification report (multi-class), and wall-clock training time in minutes. All results were serialized to `aci_comprehensive_results_with_knn.json`.

---

## CNN Architecture

The CNN treats each network flow's 78 features as a 1D sequence, allowing convolutional filters to capture local feature correlations that fully-connected layers would miss.

![CNN Architecture](Figures/CNNArch.png)
> **Figure:** A diagram of the 1D CNN architecture used in this study. The network ingests a single flow record as a sequence of shape (N, 78, 1) and passes it through three Conv1D blocks (with BatchNormalization, ReLU activation, and MaxPooling), followed by GlobalAveragePooling and two fully-connected Dense layers with dropout regularization. The architecture forks at the output into a sigmoid unit for binary classification and an 11-way softmax for multi-class classification.

```
Total parameters (binary):      67,265
Total parameters (multi-class): 67,915
Optimizer:  Adam (lr=0.001)
Batch size: 512
Early stopping patience: 4 epochs  |  LR reduce patience: 2 epochs
Binary stopped at epoch 13  |  Multi-class stopped at epoch 9
```

```python
# CNN input reshaping and training
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)
]

model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=50,
    batch_size=512,
    callbacks=callbacks
)
```

**Why 1D CNN for tabular network data?** Related network flow features (e.g., forward/backward packet stats, flag counts) tend to appear in adjacent columns, so neighboring features carry correlated information that convolution can exploit. The tradeoff is higher training time and — on this dataset — lower accuracy than tree-based methods, suggesting that the tree models' ability to learn arbitrary feature interactions is better suited to tabular intrusion detection data than fixed-kernel convolution.

---

## Models Evaluated

| Model | Type | Strengths | Use Case |
|-------|------|-----------|----------|
| **Decision Tree** | Single learner | Fast, interpretable, no hyperparameter tuning | Resource-constrained devices |
| **Random Forest** | Ensemble (bagging) | Robust, handles overfitting | Balanced accuracy-speed trade-off |
| **XGBoost** | Ensemble (boosting) | High accuracy, fast training | Production deployment |
| **KNN** | Instance-based | No training phase, simple | Baseline comparison |
| **CNN** | Deep learning (1D Conv) | Learns local feature interactions, GPU-accelerated | Deep learning baseline |

---

## Results

### Binary Classification

We first evaluated all five models on **binary classification** (Benign vs. Attack):

![Binary Classification Confusion Matrices](Figures/1_binary_confusion_matricesNEW.png)
> **Figure:** Confusion matrices for all five models on the binary classification task (Benign vs. Attack). Each matrix shows the counts of True Negatives (TN — benign correctly identified), False Positives (FP — benign flagged as attack), False Negatives (FN — attacks missed), and True Positives (TP — attacks correctly detected). Darker diagonal cells indicate stronger performance. The Decision Tree produces the fewest false positives (74), while the CNN generates the most errors overall (315 FP, 257 FN), reflecting its slightly lower accuracy on this task.

#### Analysis

All classical models achieved **>99.5% accuracy**. The CNN achieved 99.24%, slightly below the classical models:

- **Decision Tree**: Fewest false positives (74) → Best for minimizing false alarms
- **XGBoost**: Near-perfect AUC (0.9999) with fastest classical training (0.08 min)
- **Random Forest**: Comparable to XGBoost but slower training (0.31 min)
- **KNN**: Most errors among classical models
- **CNN**: 315 false positives, 257 false negatives; highest training time (1.25 min) but strong AUC of 0.9990

**Confusion Matrix Interpretation**:
- **TN (True Negative)**: Benign traffic correctly identified as benign
- **FP (False Positive)**: Benign traffic incorrectly flagged as attack (false alarm)
- **FN (False Negative)**: Attack traffic missed by model (most dangerous!)
- **TP (True Positive)**: Attack traffic correctly detected

---

### Model Comparison Tables

![Binary Model Comparison](Figures/5_binary_model_comparison.png)
> **Figure:** A summary bar chart comparing all five models across Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Training Time for the binary classification task. The y-axis is scaled to highlight the narrow performance band between models. XGBoost and Decision Tree lead on most metrics, while the CNN trails slightly on accuracy and F1 but maintains a competitive AUC of 0.9990.

#### Binary Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Time (min) |
|-------|----------|-----------|--------|----------|---------|------------|
| Decision Tree | **99.77%** | **99.87%** | 99.83% | **99.85%** | 0.9968 | 0.17 |
| Random Forest | 99.72% | 99.79% | 99.83% | 99.81% | 0.9999 | 0.31 |
| XGBoost | 99.74% | 99.82% | 99.83% | 99.82% | **0.9999** | **0.08** |
| KNN | 99.57% | 99.70% | 99.71% | 99.70% | 0.9980 | 0.00* |
| CNN | 99.24% | 99.43% | 99.53% | 99.48% | 0.9990 | 1.25 |

*KNN has no training phase (lazy learner)

**Winner**: **XGBoost** — Best balance of accuracy (99.82% F1), discrimination (0.9999 AUC), and speed (0.08 min).

**CNN Note**: While the CNN trails classical models on accuracy and F1, its AUC of 0.9990 ranks third overall, indicating strong probability calibration. The 1.25-minute training time reflects GPU overhead on a small model rather than a fundamental scalability issue.

---

![Multi-Class Model Comparison](Figures/6_multiclass_model_comparison.png)
> **Figure:** A summary bar chart comparing all five models across Accuracy, Precision, Recall, and F1-Score for the multi-class classification task (11 attack categories). The Decision Tree leads all models despite being the simplest architecture, outperforming both ensemble methods and the CNN. The CNN shows the largest gap from the classical cluster, particularly on accuracy and F1-Score, underscoring the limitations of convolutional feature extraction on hand-engineered tabular data.

#### Multi-Class Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score | Time (min) |
|-------|----------|-----------|--------|----------|------------|
| Decision Tree | **99.56%** | **99.54%** | **99.56%** | **99.55%** | **0.12** |
| Random Forest | 99.43% | 99.42% | 99.43% | 99.41% | 0.31 |
| XGBoost | 99.52% | 99.52% | 99.52% | 99.51% | 0.54 |
| KNN | 99.24% | 99.24% | 99.24% | 99.23% | 0.00* |
| CNN | 98.34% | 98.40% | 98.34% | 98.33% | 0.89 |

**Surprise Finding**: Decision Tree outperformed all models — including ensembles and the CNN — in multi-class classification.

**Why?** Multi-class problems with **distinct class boundaries** favor simpler models that create clean decision splits. Ensemble voting can blur boundaries between similar attack types (e.g., Port Scan vs. Vulnerability Scan), and the CNN's convolutional inductive bias is less effective when discriminating features are not spatially local in the feature vector.

![Multi-Class Confusion Matrices](Figures/4_multiclass_confusion_matricesNEW.png)
> **Figure:** Confusion matrices for all five models on the 11-class attack classification task. Each row represents the true attack type and each column the predicted label; diagonal cells show correct classifications. The color scale reflects sample count, making high-frequency classes (Port Scan, Benign, ICMP Flood) visually dominant. The UDP Flood row stands out across all models with near-zero recall, reflecting the severe class imbalance (only 48 test samples). The CNN matrix shows more off-diagonal spread than the tree-based models, consistent with its lower overall F1-Score.

---

### ROC-AUC Analysis

#### Full ROC Curves

![ROC Curves Full](Figures/2_roc_curves_full.png)
> **Figure:** Receiver Operating Characteristic (ROC) curves for all five models on the binary classification task, plotted across the full False Positive Rate (FPR) range of 0 to 1. Each curve shows the trade-off between True Positive Rate (TPR/Recall) and False Positive Rate at every classification threshold. All five models hug the top-left corner tightly, indicating near-perfect discrimination between benign and attack traffic. The diagonal dashed line represents a random classifier (AUC = 0.5) included as a reference baseline.

**Interpretation**: All curves hug the top-left corner, indicating **near-perfect discrimination** between benign and malicious traffic. All five models achieve AUC > 0.996.

---

#### Zoomed ROC Curves (Critical Region)

![ROC Curves Zoomed](Figures/3_roc_curves_zoomed.png)
> **Figure:** A zoomed view of the ROC curves restricted to the 0–0.1 False Positive Rate region — the operationally critical zone where a deployed IDS must minimize false alarms while maximizing attack detection. Differences between models that are invisible on the full ROC plot become apparent here. Random Forest and XGBoost reach 100% TPR at under 0.01 FPR; Decision Tree achieves 99% TPR at approximately 0.005 FPR; the CNN reaches 99% TPR around 0.008 FPR; and KNN requires the highest FPR (~0.015) to match that detection rate.

**Key Insights from Zoomed View** (0–0.1 FPR region):

1. **Random Forest & XGBoost**: Nearly identical, reaching 100% TPR at <0.01 FPR
2. **CNN**: Strong probability calibration, reaches 99% TPR around 0.008 FPR
3. **Decision Tree**: Reaches 99% TPR around 0.005 FPR
4. **KNN**: Needs ~0.015 FPR to achieve 99% TPR

---

#### AUC Score Comparison

![AUC Comparison](Figures/9_auc_comparison.png)
> **Figure:** A bar chart ranking all five models by their binary classification AUC-ROC score. The y-axis is scaled to the 0.996–1.000 range to make the small but meaningful differences between models visible. XGBoost and Random Forest tie at 0.9999, followed by the CNN (0.9990), KNN (0.9980), and Decision Tree (0.9968). A higher AUC indicates better ability to separate benign from malicious traffic across all possible classification thresholds, which is particularly important in deployments where the decision threshold must be tuned to operational requirements.

**AUC-ROC Rankings**:

1. **XGBoost**: 0.9999
2. **Random Forest**: 0.9999
3. **CNN**: 0.9990
4. **KNN**: 0.9980
5. **Decision Tree**: 0.9968

**Recommendation**: Use XGBoost for production — superior probability estimates enable **adaptive threshold tuning** based on operational requirements (e.g., prioritize recall in safety-critical scenarios).

---

### Per-Class Results

#### Per-Class Performance Breakdown

![Per-Class Performance](Figures/7_per_class_performance.png)
> **Figure:** A grouped bar chart showing per-class Precision, Recall, and F1-Score for the best-performing multi-class model (Decision Tree) across all 11 attack categories. Most classes achieve scores above 99% on all three metrics. The UDP Flood bar is a clear visual outlier — with only 48 test samples drawn from a dataset with 791 total UDP Flood records, all five models fail to learn a reliable decision boundary for this class, resulting in 29% recall and a 39% F1-Score. All other attack types are detected with high confidence.

**Per-Class Metrics (Best Multi-Class Model — Decision Tree)**:

| Attack Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Benign | 99.55% | 99.79% | 99.67% | 20,056 |
| DNS Flood | 99.86% | 99.37% | 99.61% | 2,859 |
| Dictionary Attack | 98.73% | 100.00% | 99.36% | 388 |
| ICMP Flood | 99.99% | 99.99% | 99.99% | 13,718 |
| OS Scan | 99.83% | 99.91% | 99.87% | 2,286 |
| Ping Sweep | 99.98% | 100.00% | 99.99% | 4,381 |
| Port Scan | 99.55% | 99.54% | 99.55% | 26,877 |
| SYN Flood | 100.00% | 99.76% | 99.88% | 844 |
| Slowloris | 99.74% | 99.82% | 99.78% | 1,135 |
| **UDP Flood** | **60.87%** | **29.17%** | **39.44%** | **48** |
| Vulnerability Scan | 96.03% | 95.51% | 95.77% | 2,408 |

**Critical Finding — UDP Flood Failure**:

UDP Flood is the only attack type with significantly degraded performance across all models:
- **Recall: 29.17%** — the model misses ~70% of UDP Flood attacks
- **Root cause**: Only 791 total samples in the dataset → only 48 test samples after stratified splitting
- This is a **data problem, not a model problem** — the CNN, KNN, and tree models all fail similarly on this class

All other attack types show balanced precision/recall above 95%.

---
### CNN Training Curves

![CNN Training Curves](Figures/10_cnn_training_curves.png)
> **Figure:** Training and validation loss and accuracy curves over epochs for the CNN, shown separately for the binary (left) and multi-class (right) tasks. Blue lines represent training metrics and red lines represent validation metrics. The orange shaded region on the loss panels visualizes the absolute gap between training and validation loss, serving as an overfit indicator — a widening gap would suggest the model is memorizing training data. Both tasks converge with a tight gap, indicating good generalization. Sudden drops in the loss curves mark epochs where `ReduceLROnPlateau` halved the learning rate. The binary task ran for 13 epochs and the multi-class task for 9 before early stopping triggered.

The training curve plot shows loss and accuracy over epochs for both binary and multi-class tasks, with training (blue) and validation (red) lines plotted together. The orange shaded region on the loss panels shows the absolute train/validation gap — a visual overfit indicator.

Key observations:
- **Binary** stopped at epoch 13; validation loss tracks training loss closely with minimal gap, indicating the model generalized well and did not overfit.
- **Multi-class** stopped at epoch 9; slightly faster convergence, again with a tight train/validation gap.
- Both runs used `ReduceLROnPlateau` (patience=2), which is visible as sudden drops in the loss curve where the learning rate was halved.

---

### Model Radar Chart

![Model Radar Chart](Figures/11_model_radar_chart.png)
> **Figure:** Radar (spider) charts comparing all five models simultaneously across five metrics — Accuracy, Precision, Recall, F1-Score, and AUC-ROC — for binary classification (left panel) and multi-class classification (right panel). The radial axis is zoomed to the 98–100% range to make the differences between high-performing models visible. Each model is represented by a distinct colored polygon; a larger, more outward polygon indicates better overall performance. Classical models (Decision Tree, Random Forest, XGBoost) cluster near the outer edge in both panels, while the CNN sits visibly inward on the multi-class panel, most noticeably on the Accuracy and F1-Score axes.

The radar chart plots all five models simultaneously across Accuracy, Precision, Recall, F1-Score, and AUC-ROC for both binary (left) and multi-class (right) tasks. Because all models score above 98%, the y-axis is zoomed to 98–100% to make the differences visible.

Key observations:
- Classical models (Decision Tree, Random Forest, XGBoost) cluster tightly near the outer edge in both panels.
- CNN sits slightly inward, most visibly on the multi-class panel where its 98.34% accuracy separates it from the 99%+ cluster.
- KNN traces a similar shape to the classical models in binary, but falls slightly behind in multi-class.

---

### Feature Importance Analysis

![Feature Importance](Figures/8_feature_importance.png)
> **Figure:** A horizontal bar chart showing the top 10 most important features as ranked by Random Forest's Gini impurity-based feature importances. Each bar represents a feature's relative contribution to classification decisions across all trees in the forest, expressed as a percentage of total importance. The three dominant features — RST Flag Count (28.34%), Forward Header Length (27.62%), and Source Port (23.69%) — together account for nearly 80% of all decisions. Color coding groups features by category (protocol flags, packet structure, network addressing, timing). This concentration of importance in packet-header-level features supports the case for lightweight, header-only inspection in edge deployments.

**Top 10 Most Important Features** (Random Forest importances; used as proxy for CNN, which does not expose feature importances natively):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | RST Flag Count | 28.34% | Protocol Flag |
| 2 | Fwd Header Length | 27.62% | Packet Structure |
| 3 | Src Port | 23.69% | Network Addressing |
| 4 | Bwd Packet Length Max | 5.22% | Packet Size |
| 5 | Fwd Packet Length Max | 2.76% | Packet Size |
| 6 | Connection Type | 2.18% | Network Type |
| 7 | Fwd Seg Size Min | 1.97% | Segmentation |
| 8 | Flow IAT Min | 1.74% | Timing |
| 9 | Dst IP | 1.63% | Network Addressing |
| 10 | Dst Port | 1.11% | Network Addressing |

**Top 3 features account for 79.65% of decisions**:

1. **RST Flag Count (28.3%)**: High in DDoS/port scanning; consistent in benign traffic
2. **Forward Header Length (27.6%)**: Varies by attack type — SYN Floods have small headers; data exfiltration has large headers
3. **Source Port (23.7%)**: Attackers use predictable or random high ports; benign traffic uses known service ports

**Implication**: Only packet headers need to be inspected — deep packet inspection (DPI) is not required, enabling **lightweight real-time deployment** on edge devices and V2X infrastructure.

---
# Error Analysis — Why Each Model Made Mistakes

This section provides a root-cause breakdown of model errors across both binary and multi-class classification tasks, grounded directly in the pipeline code and dataset characteristics.

---

## 0. Shared Pipeline Factors (Affects Every Model)

Before blaming any individual algorithm, several upstream decisions in the training pipeline create error conditions that every model inherits.

### 0.1 — Stratified Downsampling Crushes Rare Classes

The raw dataset has 1,231,406 samples. The pipeline downsamples to exactly 500,000:

```python
X_sampled, _, y_encoded_sampled, _, y_binary_sampled, _ = train_test_split(
    X, y_encoded, y_binary,
    train_size=500000,
    random_state=42,
    stratify=y_encoded   # proportions preserved, but rare classes get tiny counts
)
```

After the 70/15/15 split, the rarest class lands here:

| Class | Total in Dataset | After 500K Sample | In Test Set (~15%) |
|---|---|---|---|
| UDP Flood | 791 | ~321 | **~48** |
| Dictionary Attack | 6,380 | ~2,591 | ~388 |
| SYN Flood | 13,857 | ~5,627 | ~844 |

**48 test samples for UDP Flood.** No model — regardless of algorithm — can recover from this. Every model's failure on UDP Flood traces directly to this line, not to any algorithmic weakness.

---

### 0.2 — Top 3 Features Drive 79.6% of All Decisions

Feature importance from Random Forest (used as proxy for all tree models):

```python
importances = rf_multi.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False).head(10)
```

| Rank | Feature | Importance |
|---|---|---|
| 1 | RST Flag Count | 28.34% |
| 2 | Fwd Header Length | 27.62% |
| 3 | Src Port | 23.69% |
| 4–10 | All others combined | 20.35% |

**The consequence:** Any two attack classes that share similar RST Flag counts, header lengths, or port ranges will be confused with each other by every model — most critically Port Scan vs. Vulnerability Scan, and DNS Flood vs. Benign (both can use standard ports and zero RST flags).

---

### 0.3 — StandardScaler Applied Uniformly

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)    # fit only on train — correct, no leakage
X_test_scaled  = scaler.transform(X_test)
```

Tree models are scale-invariant and gain nothing from this. KNN and CNN both depend on it. However, flow statistics like `Bwd Packet Length Max` can have extreme outliers from Port Scan traffic, compressing the effective variance of other features and subtly distorting KNN's Euclidean distances.

---

## 1. Decision Tree

**Hyperparameters used:**
```python
dt_bin   = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_multi = DecisionTreeClassifier(max_depth=20, random_state=42)
```

### Binary — Why It Made Mistakes (99.77% accuracy)

The Decision Tree had the fewest false positives of any model (~74 FP), but it still errs in characteristic ways:

**False Positives (benign flagged as attack):**
RST Flag Count is the first and most powerful split (28.34% importance). Legitimate TCP connections send RST flags during teardown. A benign session with an elevated RST count — say, a client that reset several failed connections — passes the RST threshold and lands in the attack subtree. The tree has no recovery mechanism; once a flow goes down the wrong branch at depth 1, it cannot backtrack.

**False Negatives (attacks missed):**
Port Scans targeting well-known ports (80, 443, 22) have Src Port values in the benign range and minimal RST flags — they look like normal web traffic at the top-level splits. At `max_depth=20` the tree can eventually create a specific leaf for these, but only if enough training examples landed on this path. If the 70% training sample was unlucky in how these borderline flows distributed, those leaves never form.

**Why it still wins multi-class (99.56%):**
The tree's single deterministic path is actually an advantage here. It finds one clean split sequence and commits to it — no averaging, no voting, no smoothing. For classes with distinct feature profiles, this is more precise than any ensemble method.

---

### Multi-Class — Why It Made Mistakes

**Port Scan / Vulnerability Scan confusion:**
Both involve network scanning. Both show elevated RST counts. Both use varied source ports. The tree distinguishes them primarily via `Fwd Header Length`, but there is substantial overlap:

```
Port Scan:         RST > threshold → Header varies → subtree splits on Src Port range
Vulnerability Scan: RST > threshold → Header varies → same subtree, adjacent leaves
```

Flows that fall exactly on the `Fwd Header Length` boundary get assigned to whichever leaf had more training examples nearby — this produces the off-diagonal cells in the Port Scan row.

**UDP Flood (Recall: ~29%):**
~321 training samples. The tree cannot form a pure leaf for a class it barely saw. The feature profile of UDP Flood (high packet rates, specific port patterns) overlaps with ICMP Flood in the training distribution, so those flows route to the ICMP Flood leaf.

---

## 2. Random Forest

**Hyperparameters used:**
```python
rf_bin   = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_multi = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
```

### Binary — Why It Made Mistakes (99.72% accuracy, ~106 FP vs. Decision Tree's ~74)

Random Forest slightly underperforms the single Decision Tree on binary accuracy despite having a superior AUC (0.9999 vs. 0.9968). This seems counterintuitive until you understand the mechanism:

**Ensemble averaging blurs hard boundaries:**
Each of the 100 trees votes independently on a majority-vote basis. For a benign TCP flow with a moderately elevated RST count — one that the single Decision Tree's path confidently routes to benign — 40 of 100 trees trained on slightly different bootstrap samples may have seen that RST range in attack clusters and vote attack. The majority vote is benign (60–40), but the margin is slim. Flows closer to the 50–50 boundary flip to attack, becoming false positives.

```python
# Random Forest returns soft probabilities averaged across all trees
rf_proba = rf_bin.predict_proba(X_test_scaled)[:, 1]
# Threshold at 0.5 — flows at 0.51 attack probability are classified as attack
# The Decision Tree has harder, more committed probabilities per leaf
```

**Why AUC is better despite higher FP rate:**
The averaged probabilities across 100 trees are far better calibrated than any single tree's leaf probability. This makes Random Forest better at probability ranking (AUC) but not necessarily better at hard threshold classification (accuracy at 0.5).

---

### Multi-Class — Why It Made Mistakes (99.43%, worse than single Decision Tree)

**Bootstrap sampling destroys UDP Flood:**
Each tree in the forest is trained on a bootstrap sample (sampling with replacement). With only ~321 UDP Flood records in 500,000 total, the probability that a given tree's bootstrap sample contains at least one UDP Flood example is ~47%. This means roughly **53 of 100 trees have zero UDP Flood training examples** — those trees have no UDP Flood leaf and absorb UDP Flood test samples into the nearest cluster.

```python
# Conceptually, a tree trained on a bootstrap with no UDP Flood:
# → No leaf for UDP Flood
# → UDP Flood test samples fall into nearest leaf (ICMP Flood, Port Scan, etc.)
# → 100-tree vote: 53 trees say "not UDP Flood", 47 say "UDP Flood"
# → Majority: wrong class
```

**DNS Flood / Benign confusion amplified by voting:**
DNS Flood uses UDP port 53, produces valid-looking DNS queries, and generates zero RST flags. Some trees split on RST first and route DNS Flood to the Benign subtree. In the single Decision Tree, a later split on `Flow IAT Min` (timing) corrects this. In Random Forest, trees that didn't get enough DNS Flood / Benign boundary examples never learn that later split — those trees consistently vote Benign for DNS Flood, and the ensemble accumulates that error.

---

## 3. XGBoost

**Hyperparameters used:**
```python
# Binary
xgb_bin = xgb.XGBClassifier(
    n_estimators=100, max_depth=10, learning_rate=0.1,
    random_state=42, n_jobs=-1, eval_metric='logloss'
)

# Multi-Class
xgb_multi = xgb.XGBClassifier(
    n_estimators=100, max_depth=10, learning_rate=0.1,
    random_state=42, n_jobs=-1,
    objective='multi:softmax',   # ← important: hard labels, no probabilities
    num_class=num_classes,
    eval_metric='mlogloss'
)
```

### Binary — Why It Made Mistakes (99.74%, AUC: 0.9999)

XGBoost is the best production model overall, but its errors have a specific character rooted in how gradient boosting works:

**`learning_rate=0.1` is aggressive for borderline flows:**
Gradient boosting trains sequentially — each new tree fits the residuals of all previous trees. At `lr=0.1`, the model updates quickly on hard examples. Borderline benign flows that early trees misclassify as attack get high residuals. Later trees overfit to those specific flows, creating fine-grained attack predictions that may not generalize to similar-but-different test flows. A lower `learning_rate` (0.01–0.05) with more estimators would smooth this out.

**`max_depth=10` vs. Decision Tree's `max_depth=20`:**
XGBoost trees are shallower (10 vs. 20). Individual trees capture coarser patterns. The boosting compensates via iteration, but for very specific boundary cases — like a Port Scan on port 443 with a standard header — the depth-10 tree may not create the specific leaf needed, leaving that correction to later boosting rounds that are increasingly focused on other errors.

---

### Multi-Class — Why It Made Mistakes (99.52%)

**`multi:softmax` outputs hard labels with no probability calibration:**
This is the most impactful code-level choice. `multi:softmax` forces XGBoost to output a single integer class per sample rather than class probabilities:

```python
# multi:softmax → hard labels, no calibration
xgb_multi_pred = xgb_multi.predict(X_test_scaled)   # integer class IDs

# Compare to binary mode which outputs probabilities:
xgb_proba = xgb_bin.predict_proba(X_test_scaled)[:, 1]  # calibrated [0, 1]
```

Using `multi:softprob` instead would produce class probabilities and allow threshold tuning, likely improving boundary cases between Port Scan and Vulnerability Scan.

**Boosting focuses on the easy majority first:**
In the early boosting rounds, Port Scan (35.8% of data) and Benign (26.7%) dominate the loss function. The model spends most of its capacity getting these right. By round 100, when it should be refining minority class boundaries, the remaining learning rate budget is small and the corrections are coarse. UDP Flood's contribution to the total loss is ~0.06% of the dataset — it never meaningfully moves the loss function regardless of how wrong it is.

---

## 4. K-Nearest Neighbors (KNN)

**Hyperparameters used:**
```python
knn_bin   = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_multi = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# Distance metric: Euclidean (default)
# Feature weighting: uniform (default) — all features weighted equally
```

### Binary — Why It Made Mistakes (99.57%, most errors among classical models)

KNN's errors are structurally different from tree models. The algorithm has no concept of feature importance — every feature contributes equally to the Euclidean distance calculation.

**All 47 features are weighted identically:**
```python
# KNN computes: distance = sqrt(sum((x_i - x_j)^2 for all features))
# RST Flag Count (28.34% importance) has EQUAL weight to Dst IP (1.63% importance)
# A flow with a very different RST count but matching values on 44 irrelevant
# features appears CLOSE in Euclidean space — and gets the wrong label.
```

**`k=5` in high-dimensional space (47 features):**
In high dimensions, Euclidean distance becomes less meaningful — the ratio of max to min pairwise distance approaches 1 as dimensions increase. Two flows can be very different in the 3 features that matter (RST Flag, Header Length, Src Port) but appear close because they match on 44 irrelevant features. The `k=5` neighborhood pulls in flows that are genuinely dissimilar in discriminative feature space.

**Port Scan density dominates neighborhoods:**
Port Scan has ~178,000 records in the 500K sample (35.8%). In Euclidean space, borderline flows near the benign/attack boundary have a high probability of having Port Scan flows among their 5 nearest neighbors purely due to class density — even if the query flow is genuinely benign.

---

### Multi-Class — Why It Made Mistakes (99.24%)

**No feature importance weighting in the distance metric:**
Tree models discover that RST Flag Count is the most important feature and route flows through it first. KNN gives RST Flag Count the same weight as `Fwd Seg Size Min` (1.97% importance). This means a Vulnerability Scan flow and a Port Scan flow with identical RST counts but different segment sizes will appear equidistant from a Port Scan query flow — the neighborhood is noise-contaminated.

**Port Scan density swamps minority class neighborhoods:**
```python
# With ~178,000 Port Scan training samples, any borderline multi-class flow
# is statistically likely to have Port Scan neighbors.
# k=5: even 1 wrong neighbor out of 5 shifts the vote.
# For Vulnerability Scan: if 3 of 5 neighbors are Port Scan → misclassified.
```

**UDP Flood:** Only ~321 training samples. The 5 nearest neighbors of any UDP Flood test flow are almost certainly from the nearest large class (ICMP Flood or Port Scan). KNN cannot overcome this — it has no mechanism to weight rare classes higher or to learn a global decision boundary.

**Slowloris:** Slow-rate attacks are designed to look like legitimate slow connections. Their flow timing statistics (low packet rate, small sizes, long duration) place them in the same Euclidean neighborhood as genuine slow HTTP sessions. KNN has no way to distinguish these without a feature that specifically encodes "partial request never completed" — which is not captured in the 78 flow-level statistics.

---

## 5. 1D CNN (Deep Learning Baseline)

**Architecture and training:**
```python
def build_cnn(output_units, output_activation, loss, n_features, lr=0.001):
    inp = keras.Input(shape=(n_features, 1))            # (batch, 78, 1)

    x = layers.Conv1D(64,  kernel_size=3, padding='same')(inp)  # kernel sees 3 adjacent features
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)             # → (batch, 39, 64)

    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)             # → (batch, 19, 128)

    x = layers.Conv1D(64,  kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling1D()(x)              # → (batch, 64) — positional info LOST
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,  activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(output_units, activation=output_activation)(x)
    ...

# Input reshaping:
X_train_cnn = X_train_scaled.reshape(-1, n_features, 1)
# Each row of 78 features is treated as a 1D signal of length 78
```

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]
# Binary:     stopped at epoch 13 of 50
# Multi-class: stopped at epoch 9 of 25
```

### Binary — Why It Made Mistakes (99.24%, 315 FP, 257 FN)

**The core assumption is wrong:**
`Conv1D` with `kernel_size=3` assumes that features at positions N, N+1, and N+2 are correlated — the same way adjacent pixels in an image or adjacent timesteps in a time series are correlated. For ACI-IoT-2023, the 78 features are independently computed flow statistics. The feature at column 5 (e.g., `Fwd Header Length`) has no inherent relationship to column 6 or column 7 — they were placed in that order by the CSV export, not because they are spatially or temporally adjacent.

```
Feature index:  0         1         2         3         4         5
Feature name:  duration  protocol  src_port  dst_port  pkt_len   header_len
                                                        ↑
                         Conv1D kernel sees [pkt_len, header_len, ???]
                         and tries to learn a pattern from this arbitrary grouping
```

The kernels learn correlations between arbitrarily adjacent columns — a meaningless inductive bias for this data.

**`GlobalAveragePooling1D` destroys discriminative signals:**
After the convolutional layers reduce the sequence from 78 → 39 → 19 positions, `GlobalAveragePooling1D` averages all 19 remaining positions into a single 64-dimensional vector:

```python
x = layers.GlobalAveragePooling1D()(x)
# Output: mean across all positions
# RST Flag Count (most important) is averaged together with Dst IP (least important)
# The specific value of RST Flag Count — the single most predictive feature — is
# diluted by averaging with 63 other feature-position activations.
```

Tree models split on RST Flag Count first and alone. The CNN buries it in a global average.

**315 False Positives:**
Benign flows with a moderately elevated RST count (legitimate TCP teardowns) produce ambiguous sigmoid outputs clustered around 0.45–0.55. The threshold is applied at 0.5 without tuning:

```python
cnn_bin_pred = (cnn_bin_proba >= 0.5).astype(int)
```

Flows that tree models confidently route to a benign leaf (RST elevated but header and port both benign) get sigmoid scores of 0.52 from the CNN — because the convolutional filters learned averaged correlations, not the crisp boundary defined by RST alone. Those flows cross the 0.5 line and become false positives.

**257 False Negatives:**
Attacks that don't exhibit strong local feature correlations — particularly Slowloris (timing features spread across distant columns) and Port Scan variants targeting common ports — produce attack probabilities below 0.5 because no single kernel window captures a discriminative signal. The CNN cannot jointly consider features at positions 1, 25, and 72 the way a tree's decision path can.

**Stopped at epoch 13, not from overfitting:**
```python
EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
```
The CNN did not stop because it was overfitting — the train/val loss gap is tight throughout training. It stopped because it hit a performance ceiling imposed by the architectural mismatch. There is no more signal available to the convolutional architecture regardless of how many more epochs it trains.

---

### Multi-Class — Why It Made Mistakes (98.34% — worst of all 5 models)

The gap between CNN and all other models widens from ~0.5% in binary to ~1.2% in multi-class. Fine-grained class discrimination amplifies every architectural weakness.

**Stopped at epoch 9, faster plateau than binary:**
```python
# Binary:      stopped at epoch 13
# Multi-class: stopped at epoch 9
```
The multi-class task hits its ceiling faster because the convolutional representation is even less suited to distinguishing 11 specific attack types than it is to separating attack from benign in aggregate.

**Port Scan dominates the learned weight space:**
Port Scan is 35.8% of the data. During training, the CNN's filters are updated primarily by Port Scan gradients. The convolutional kernels effectively become "Port Scan detectors." At inference time, the softmax assigns elevated Port Scan probability to any flow that partially matches the Port Scan filter response — including Vulnerability Scans, OS Scans, and even some Benign flows. This explains the off-diagonal spread visible in the Port Scan column of the CNN confusion matrix.

**`Dropout(0.3)` is too aggressive for 11-class fine-grained decisions:**
```python
x = layers.Dropout(0.3)(x)   # drops 30% of Dense activations during training
```
For binary classification, Dropout(0.3) is a reasonable regularizer. For 11-class softmax where minority classes (UDP Flood, Dictionary Attack, SYN Flood) need precise probability estimates, dropping 30% of Dense activations at each training step introduces calibration noise that degrades the softmax for rare classes. A lower rate (0.1–0.15) would likely improve multi-class performance.

**The fundamental architectural mismatch (summary):**

| Assumption Required for CNN to Work | Reality of ACI-IoT-2023 |
|---|---|
| Features N and N+1 are correlated | Features are independent flow statistics |
| Pattern location is irrelevant (translation invariance) | Feature position has no semantic meaning at all |
| Low-level features combine hierarchically | All 78 features are already high-level statistics |
| Global average captures class identity | Averaging destroys the specific feature values that define each class |

Tree models succeed because they find **arbitrary combinations of any features at any positions** via recursive splitting — exactly what tabular NIDS data requires. The CNN's fixed-kernel, local-convolution approach cannot replicate this, and no amount of hyperparameter tuning changes that fundamental constraint.

---

## 6. Cross-Cutting Error Patterns

### 6.1 — UDP Flood: Universal Failure Across All Models

Every model fails on UDP Flood. The cause is identical for all five:

| Model | UDP Flood Recall | Why |
|---|---|---|
| Decision Tree | ~29% | Cannot form a pure leaf from ~224 training samples |
| Random Forest | ~25% | ~53 of 100 trees have zero UDP Flood bootstrap samples |
| XGBoost | ~28% | UDP Flood contributes <0.06% to total loss; ignored by boosting |
| KNN | ~30% | 5 nearest neighbors dominated by majority class neighbors |
| CNN | ~20% | Filters never learned UDP Flood pattern; softmax ≈ 1/11 |

**The fix is in the data, not the model:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train_multi)
# Generating even 5,000 synthetic UDP Flood samples would give all models
# a learnable signal before the split happens.
```

---

### 6.2 — Port Scan / Vulnerability Scan: Shared Feature Profile

This is the most commonly confused class pair in multi-class classification across all models. Both classes exhibit:
- Elevated RST Flag Counts (scanning generates resets)
- Variable Fwd Header Length (different scan types)
- High Src Port entropy (randomized source ports)

The only meaningful distinction — destination port range and connection success rate — is not among the top-10 most important features. All models that use feature-importance-based splitting (tree models) never prioritize these features early enough, and KNN/CNN weight them equally with noise features.

---

### 6.3 — Why Binary Always Outperforms Multi-Class

```
Binary:      Benign vs. {Port Scan, Vulnerability Scan, ICMP Flood, DNS Flood, ...}
Multi-class: Benign vs. Port Scan vs. Vulnerability Scan vs. ICMP Flood vs. ...
```

In binary mode, the model only needs RST Flag Count > threshold → attack. The entire confused Port Scan / Vulnerability Scan pair collapses into one label. In multi-class, the model must distinguish those two from each other using features that don't cleanly separate them.

The CNN's gap is largest (99.24% binary → 98.34% multi-class, a **0.9% drop**) while classical models drop only 0.1–0.3%. This confirms the architectural mismatch: the CNN's global averaging approach is barely adequate for the coarse binary problem and breaks down under the finer discrimination required by 11 classes.

---

## 7. Summary: Root Cause by Model and Task

| Model | Binary Root Cause | Multi-Class Root Cause | Unique Structural Weakness |
|---|---|---|---|
| **Decision Tree** | RST threshold catches legitimate TCP teardowns (FP); Port Scans on known ports miss RST filter (FN) | Port Scan/Vulnerability Scan boundary overlap; UDP Flood scarcity | Axis-aligned splits cannot capture diagonal feature boundaries |
| **Random Forest** | 100-tree voting averages over disagreement on borderline flows, adding ~32 extra FP vs. Decision Tree | Bootstrap sampling leaves ~53% of trees with zero UDP Flood examples | Ensemble averaging is a disadvantage when class boundaries are sharp |
| **XGBoost** | `lr=0.1` overcorrects on borderline flows in later boosting rounds | `multi:softmax` gives no probability calibration; majority class dominates early boosting | Fast learning rate; softmax mode loses calibration for minority classes |
| **KNN** | All 47 features weighted equally; RST Flag Count carries same distance weight as irrelevant features | Port Scan density (~178K records) dominates 5-nearest-neighbor votes for borderline flows | No feature importance; uniform distance weighting in high-dimensional space |
| **CNN** | `kernel_size=3` finds correlations between arbitrarily adjacent features; `GlobalAveragePooling` dilutes RST signal | `Dropout(0.3)` too aggressive for 11-class calibration; architecture cannot jointly evaluate non-adjacent features | Inductive bias (local spatial correlation) is fundamentally wrong for tabular flow data |
---
## Key Findings

### 1. Classical Models Outperform Deep Learning for Tabular NIDS

The CNN (98.34% multi-class F1) underperforms all classical models (99.23%–99.55%). For structured, tabular network flow data with strong feature-level signals, tree-based methods are more appropriate than convolutional architectures. The CNN's inductive bias for local correlations does not translate well to hand-engineered flow statistics.

### 2. XGBoost is Optimal for Production

- **Best AUC-ROC**: 0.9999 (binary)
- **Fast training**: 0.08 min (binary), 0.54 min (multi-class)
- **Near-optimal F1**: 99.82% binary, 99.51% multi-class
- **Robust**: Ensemble method resistant to overfitting

### 3. Decision Tree Excels in Multi-Class

Simple Decision Tree outperformed all models in multi-class classification (99.56% vs 98.34% CNN, 99.52% XGBoost). Multi-class problems with distinct attack signatures benefit from clear decision boundaries rather than complex ensemble voting or convolutional feature extraction.

### 4. Class Imbalance is a Critical Challenge

**UDP Flood detection failed (29% recall)** across all models due to only 0.06% dataset representation. It is a data problem.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 5. Protocol-Level Features Are Sufficient

79.6% of classification decisions rely on only 3 features. Lightweight deployment is possible without deep packet inspection.

### 6. Training Efficiency Enables Edge Deployment

| Model | Binary Time | Multi-Class Time | Suitable For |
|-------|-------------|-----------------|--------------|
| XGBoost | 0.08 min | 0.54 min | Real-time retraining on edge devices |
| Decision Tree | 0.17 min | 0.12 min | Embedded systems (vehicles, RSUs) |
| Random Forest | 0.31 min | 0.31 min | Cloud-based batch processing |
| KNN | ~0.00 min* | ~0.00 min* | Baseline/reference systems |
| CNN | 1.25 min | 0.89 min | GPU-accelerated environments |

*KNN has no training phase (lazy learner)

---

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/connorgladish/IoTResearch.git
cd IoTResearch

# Install requirements
pip install -r requirements.txt
```

---

## Project Structure

```
IoTResearch/
├── data/
│   └── ACI-IoT-2023.csv                          # Raw dataset (download separately via link in repo)
├── Results/
│   ├── aci_comprehensive_results_with_knn.json   # Full results (all 5 models incl. CNN)
│   └── aci_comprehensive_results.json            # Results without KNN
├── ColabRun/
│   ├── ColabKaggleGDriveLoad.py
│   ├── ColabKaggleLink.py
│   ├── ColabTrainingScript.py
│   ├── ColabVisualizationGenerator.py
│   └── README.md
├── Figures/
│   ├── 1_binary_confusion_matrices.png
│   ├── 2_roc_curves_full.png
│   ├── 3_roc_curves_zoomed.png
│   ├── 4_multiclass_confusion_matrices.png
│   ├── 5_binary_model_comparison.png
│   ├── 6_multiclass_model_comparison.png
│   ├── 7_per_class_performance.png
│   ├── 8_feature_importance.png
│   └── 9_auc_comparison.png
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
└── LICENSE                                       # MIT License
```

## Disclaimer

This research was done utilizing Google Colab, so local downloads were not tested. Therefore, some directory structures that are shown may not work exactly as pictured. Please play around with the structure to make it work for you!
