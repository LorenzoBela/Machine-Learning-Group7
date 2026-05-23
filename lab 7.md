Adamson University Computer Engineering Department

**Laboratory Exercise 7**

**Model Retraining and Multi-Model Performance Comparison**

Submitted by:

**Group #**

| **Category**                                               | **Exceptional**<br><br>**4**                                                                                                                                                    | **Acceptable**<br><br>**3**                                                                   | **Marginal**<br><br>**2**                                                                                    | **Unacceptable**<br><br>**1**                                                   | **Score** |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | --------- |
| **System / Pipeline Design & Implementation (30%)**        | Clear, well-structured machine learning pipeline or experimental design that fully meets the stated objectives, requirements, and constraints of the lab.                       | Adequate pipeline or experimental design with minor limitations; meets most lab requirements. | Partial or loosely structured design; some requirements addressed but key elements are missing or incorrect. | Minimal or unclear design effort; does not address the lab requirements.        |           |
| **Application of Tools & Techniques**<br><br>**(25%)**     | Correct selection and expert use of appropriate tools and techniques (e.g., Python, Jupyter, ML libraries, data analysis tools); methods are effectively applied and justified. | Correct tool selection with minor errors or inconsistencies in application.                   | Limited, inappropriate, or incorrect tool usage; techniques partially support the task.                      | No meaningful or incorrect use of required tools and techniques.                |           |
| **Implementation & Resource Utilization**<br><br>**(20%)** | Efficient, logical, and well-organized implementation; methods and resources are fully aligned with the problem and constraints.                                                | Functional implementation with minor inefficiencies or redundancies.                          | Implementation partially works but lacks efficiency, clarity, or completeness.                               | Poor or non-functional implementation with little consideration of constraints. |           |
| **Testing, Analysis & Validation**<br><br>**(15%)**        | Comprehensive testing and analysis; results are clearly validated, interpreted, and supported by appropriate metrics, figures, or tables.                                       | Adequate testing and analysis with mostly correct interpretation of results.                  | Limited testing; analysis is incomplete, weakly supported, or partially incorrect.                           | No testing performed or results are incorrectly analyzed or interpreted.        |           |
| **Documentation & Reporting**<br><br>**(10%)**             | Clear, complete, and well-structured lab report/notebook with proper figures, tables, explanations, and reflection.                                                             | Complete documentation with minor issues in clarity, organization, or detail.                 | Partial documentation; missing sections, unclear explanations, or poor organization.                         | Incomplete, poorly written, or missing documentation.                           |           |
| **TOTAL SCORE**                                            |                                                                                                                                                                                 |                                                                                               |                                                                                                              |                                                                                 |           |

| **Group Members** | | | |
| --- | | | | --- | --- | --- |
| **STUDENT NUMBER** | **NAME** | **CONTRIBUTION** | **SCORE** |
| | | | |
| | | | |
| | | | |
| | | | |

Submitted to:

Engr. Dexter James L. Cuaresma

Date:

mm/dd/year

**OBJECTIVES**

- Justify the selection of four machine learning models appropriate for a given classification dataset.
- Re-train all four models on the group's existing dataset using consistent preprocessing and data split procedures.
- Evaluate and compare model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Identify overfitting, underfitting, and generalization differences across the four trained models.
- Recommend the most suitable model for deployment based on empirical performance evidence and computational cost.

**DISCUSSION**

# **Introduction**

In Laboratory Exercise 6, students designed and evaluated three Convolutional Neural Network (CNN) architectures on a custom image dataset. While CNNs are powerful for spatial data, real-world machine learning practice requires engineers and data scientists to benchmark multiple algorithm families before committing to a final system. Different models carry different inductive biases, computational demands, and generalization behaviors - and the best model for one dataset may perform poorly on another.

Laboratory Exercise 7 extends this comparative analysis to a broader model selection study. Each group will select four distinct machine learning or deep learning models, retrain them on the same dataset used in Lab 6, and systematically compare their performance. The models may span classical machine learning approaches (e.g., Support Vector Machines, Random Forests, k-NN), shallow neural networks (MLPs), or alternative deep learning architectures (e.g., transfer learning with pretrained CNNs, EfficientNet, MobileNet).

By the end of this exercise, students will have built practical experience in the complete model selection pipeline: hypothesis (model choice rationale), experiment (training and evaluation), and decision (model recommendation) - a cycle central to every applied ML project.

# **Detailed Discussion**

## **1\. The Model Selection Problem**

No single algorithm universally outperforms all others on every dataset - a result formalized in the No Free Lunch Theorem (Wolpert, 1996). The theorem implies that model choice must be empirically justified for each specific problem. The key dimensions along which models differ include:

- **Inductive bias:** assumptions baked into the model structure (e.g., CNNs assume spatial locality; SVMs assume a separating hyperplane).
- **Capacity and complexity:** the number of learnable parameters and the function class the model can represent.
- **Computational cost:** training time, inference latency, and memory footprint.
- **Generalization behavior:** sensitivity to overfitting, regularization methods, and performance on unseen data.

## **2\. Overview of Model Families**

**a) Classical Machine Learning Models**

Classical models such as Support Vector Machines (SVM), Random Forests, Gradient Boosting (XGBoost, LightGBM), and k-Nearest Neighbors (k-NN) operate on feature vectors. When applied to image data, raw pixels or CNN-extracted embeddings are typically used as input features. These models train quickly and often provide strong baselines, particularly on small datasets.

**b) Multilayer Perceptrons (MLPs)**

Fully connected networks flatten the input and apply sequences of linear transformations with non-linear activations. While MLPs lack spatial inductive bias, they are universal function approximators and serve as important reference points for understanding how architecture affects performance.

**c) Transfer Learning with Pretrained CNNs**

Transfer learning leverages feature representations learned on large datasets (e.g., ImageNet) by freezing or fine-tuning the convolutional backbone of models such as ResNet-18, MobileNetV2, or EfficientNet-B0. This is particularly effective when the group dataset is small, as it dramatically reduces the risk of overfitting while providing high-quality feature extraction.

**d) Custom CNN Architectures (from Lab 6)**

Groups may choose to include one or more of their Lab 6 CNN models as a baseline or reference point. In such cases, the model must be retrained from scratch using the same pipeline defined in this exercise (do not reuse Lab 6 training results).

## **3\. Performance Metrics for Multi-Model Comparison**

Since the dataset may be imbalanced (unequal class frequencies), accuracy alone is insufficient. Each model must be evaluated on all of the following metrics:

- **Test Accuracy:** fraction of correctly classified test samples.
- **Precision (macro-averaged):** positive predictive value per class, averaged without regard to class size.
- **Recall (macro-averaged):** true positive rate per class, averaged across classes.
- **F1-Score (macro-averaged):** harmonic mean of precision and recall; robust to class imbalance.
- **Confusion Matrix:** per-class breakdown of correct and incorrect predictions; reveals error patterns.
- **Training Time:** wall-clock time to complete training; important for resource planning.
- **Parameter Count (for neural models):** total trainable parameters; proxy for model capacity.

## **4\. The Four Models in This Lab**

Each group must select four models. At least one must be from the classical ML family (non-deep-learning), and at least one must be a neural network. The table below should be completed by each group before beginning implementation:

| Model   | Architecture / Name | Category     | Rationale for Selection                          |
| ------- | ------------------- | ------------ | ------------------------------------------------ |
| Model A | Group-Chosen #1     | TBD by group | First model selected and justified by the group  |
| Model B | Group-Chosen #2     | TBD by group | Second model selected and justified by the group |
| Model C | Group-Chosen #3     | TBD by group | Third model selected and justified by the group  |
| Model D | Group-Chosen #4     | TBD by group | Fourth model selected and justified by the group |

_Note: Groups are encouraged to choose models that offer diverse architectural perspectives. Avoid selecting four models from the same family (e.g., four CNNs). Diversity in model type leads to more informative comparisons._

**MATERIALS**

# **Requirements**

## **Hardware**

- Laptop/PC with at least 8 GB RAM (GPU recommended for deep learning models)
- Google Colab (free tier or Colab Pro) is acceptable for GPU-accelerated training

## **Software**

- Python 3.10+
- Jupyter Notebook / Google Colab

## **Libraries**

- torch, torchvision (for neural network models)
- scikit-learn (for classical ML models, metrics, cross-validation)
- numpy, pandas
- matplotlib, seaborn
- timm (optional, for pretrained vision models)
- PIL (Pillow)

## **Dataset / Data Source**

- Same group-collected image dataset used in Laboratory Exercise 6
- Dataset must be organized per class in subfolders (ImageFolder format)

**PROCEDURES**

# **Procedures**

## **Part A) Environment and Dataset Setup**

- Create a directory: ml-perception-labs/lab07_model_comparison/
- Create the following folder structure:

lab07_model_comparison/

├── data/

│ └── raw/ - same dataset from Lab 6

├── notebook/

│ └── Lab07_ModelComparison.ipynb

├── models/ - saved model weights (.pth or .pkl)

└── outputs/

├── figures/

└── tables/

- Create a notebook named: notebook/Lab07_ModelComparison.ipynb
- In the first notebook cell, display: Name, Section, Date, Dataset name, and the four selected models
- Install and import all required libraries (Cell 1):

\# Cell 1 --- Install and import dependencies

!pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow timm

import torch, torchvision, numpy as np, matplotlib.pyplot as plt

import seaborn as sns, pandas as pd, time

from pathlib import Path

from sklearn.metrics import (confusion_matrix, classification_report,

accuracy_score, precision_score,

recall_score, f1_score)

from sklearn.model_selection import cross_val_score

print(f"PyTorch : {torch.\__version_\_}")

print(f"CUDA : {torch.cuda.is_available()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")

## **Part B) Load and Prepare the Dataset**

Reuse the same dataset from Lab 6. Apply the same preprocessing pipeline to ensure a fair comparison across all four models.

- Reuse the same 70% train / 15% val / 15% test split with the same random seed (SEED = 42):

\# Cell 2 --- Dataset loading and splitting (same as Lab 6)

import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader, random_split

IMG_SIZE = 64

BATCH_SIZE = 32

SEED = 42

train_transform = transforms.Compose(\[

transforms.Resize((IMG_SIZE, IMG_SIZE)),

transforms.RandomHorizontalFlip(p=0.5),

transforms.RandomRotation(degrees=15),

transforms.ColorJitter(brightness=0.2, contrast=0.2),

transforms.ToTensor(),

transforms.Normalize(mean=\[0.485, 0.456, 0.406\],

std=\[0.229, 0.224, 0.225\]),

\])

eval_transform = transforms.Compose(\[

transforms.Resize((IMG_SIZE, IMG_SIZE)),

transforms.ToTensor(),

transforms.Normalize(mean=\[0.485, 0.456, 0.406\],

std=\[0.229, 0.224, 0.225\]),

\])

DATA_DIR = Path("../data/raw")

full_dataset = ImageFolder(root=DATA_DIR, transform=train_transform)

CLASS_NAMES = full_dataset.classes

NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

n_total = len(full_dataset)

n_train = int(0.70 \* n_total)

n_val = int(0.15 \* n_total)

n_test = n_total - n_train - n_val

train_set, val_set, test_set = random_split(

full_dataset, \[n_train, n_val, n_test\],

generator=torch.Generator().manual_seed(SEED)

)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,

shuffle=True, num_workers=2)

val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,

shuffle=False, num_workers=2)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,

shuffle=False, num_workers=2)

print(f"Train: {len(train_set)} Val: {len(val_set)} Test: {len(test_set)}")

- For classical ML models (SVM, Random Forest, etc.), flatten the images into feature vectors and store them as NumPy arrays (Cell 3):

\# Cell 3 --- Extract flat feature arrays for classical ML models

def extract_features(loader):

features, labels = \[\], \[\]

for imgs, lbls in loader:

features.append(imgs.view(imgs.size(0), -1).numpy())

labels.append(lbls.numpy())

return np.concatenate(features), np.concatenate(labels)

X_train, y_train = extract_features(train_loader)

X_val, y_val = extract_features(val_loader)

X_test, y_test = extract_features(test_loader)

print(f"X_train shape: {X_train.shape}")

print(f"X_test shape: {X_test.shape}")

## **Part C) Define and Implement All Four Models**

Implement each of the four selected models in separate, clearly labeled cells. Include inline comments explaining key design decisions, hyperparameter choices, and any preprocessing specific to that model.

- **Cell 4 - Model A:** First selected model. Define architecture or instantiate from sklearn/timm. Include parameter count or model summary.
- **Cell 5 - Model B:** Second selected model.
- **Cell 6 - Model C:** Third selected model.
- **Cell 7 - Model D:** Fourth selected model.

A reference template for a deep learning model (Cell 4 example):

\# Cell 4 --- Model A: \[Group-chosen architecture name\]

\# Rationale: \[Explain why this model was chosen\]

import torch.nn as nn

import torch.nn.functional as F

class ModelA(nn.Module):

"""\[Brief description of the architecture\]"""

def \__init_\_(self, num_classes):

super().\__init_\_()

\# Define layers here

...

def forward(self, x):

\# Define forward pass

...

model_a = ModelA(NUM_CLASSES).to(DEVICE)

params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)

print(f"Model A | Parameters: {params_a:,}")

A reference template for a classical ML model:

\# Cell 5 --- Model B: \[e.g., Support Vector Machine\]

\# Rationale: \[Explain why this model was chosen\]

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

model_b = Pipeline(\[

('scaler', StandardScaler()),

('svm', SVC(kernel='rbf', C=1.0, gamma='scale',

probability=True, random_state=SEED))

\])

print("Model B: SVM (RBF kernel) - no learnable parameter count for sklearn models")

## **Part D) Define Training Pipelines**

Define separate training routines for neural network models and classical ML models. Record training time for all models.

- Use the following reusable training loop for all neural network models (Cell 8):

\# Cell 8 --- Reusable neural network training loop

import torch.optim as optim

NUM_EPOCHS = 30

LR = 1e-3

def make_optimizer(model):

return optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

def make_scheduler(optimizer):

return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, val_loader, num_epochs, model_name):

optimizer = make_optimizer(model)

scheduler = make_scheduler(optimizer)

history = {'train_loss':\[\], 'val_loss':\[\], 'train_acc':\[\], 'val_acc':\[\]}

best_val_acc, best_weights = 0.0, None

start_time = time.time()

for epoch in range(1, num_epochs + 1):

model.train()

run_loss, correct, total = 0.0, 0, 0

for imgs, labels in train_loader:

imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

optimizer.zero_grad()

out = model(imgs)

loss = criterion(out, labels)

loss.backward()

optimizer.step()

run_loss += loss.item() \* imgs.size(0)

correct += out.max(1)\[1\].eq(labels).sum().item()

total += labels.size(0)

train_loss = run_loss / total

train_acc = correct / total

model.eval()

v_loss, v_correct, v_total = 0.0, 0, 0

with torch.no_grad():

for imgs, labels in val_loader:

imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

out = model(imgs)

v_loss += criterion(out, labels).item() \* imgs.size(0)

v_correct += out.max(1)\[1\].eq(labels).sum().item()

v_total += labels.size(0)

val_loss = v_loss / v_total

val_acc = v_correct / v_total

scheduler.step()

for k, v in zip(\['train_loss','val_loss','train_acc','val_acc'\],

\[train_loss, val_loss, train_acc, val_acc\]):

history\[k\].append(v)

if val_acc > best_val_acc:

best_val_acc = val_acc

best_weights = {k: v.clone() for k, v in model.state_dict().items()}

if epoch % 5 == 0 or epoch == 1:

print(f"\[{model_name}\] Ep {epoch:3d} | "

f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} | "

f"ValLoss={val_loss:.4f} Acc={val_acc:.4f}")

model.load_state_dict(best_weights)

elapsed = time.time() - start_time

print(f" Best Val Acc ({model_name}): {best_val_acc:.4f} | Time: {elapsed:.1f}s")

return history, elapsed

- Use the following training block for classical ML models (Cell 9):

\# Cell 9 --- Classical ML training wrapper

def train_classical(model, X_train, y_train, model_name):

start_time = time.time()

model.fit(X_train, y_train)

elapsed = time.time() - start_time

print(f"\[{model_name}\] Training complete | Time: {elapsed:.1f}s")

return elapsed

## **Part E) Train All Four Models**

Train each model using its appropriate pipeline. Record training time. Save all trained model weights to the models/ directory (Cell 10):

\# Cell 10 --- Train all four models

MODELS_DIR = Path("../models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

\# --- Neural network models ---

print("=" \* 60)

print("Training Model A ---", MODEL_A_NAME)

history_a, time_a = train_model(model_a, train_loader, val_loader,

NUM_EPOCHS, MODEL_A_NAME)

torch.save(model_a.state_dict(), MODELS_DIR / "model_a.pth")

\# Repeat for Model B, C, D as needed

\# For classical models, call train_classical() instead

\# e.g.:

\# time_b = train_classical(model_b, X_train, y_train, MODEL_B_NAME)

\# import joblib; joblib.dump(model_b, MODELS_DIR / 'model_b.pkl')

Record training loss, validation loss, training accuracy, and validation accuracy per epoch for all neural network models. Classical models do not produce epoch-level history - record only final training time.

## **Part F) Evaluate All Four Models**

Evaluate every model on the held-out test set. Compute all required metrics and generate visualizations. Save all figures to outputs/figures/ and tables to outputs/tables/.

- Plot training curves (loss and accuracy over epochs) for all neural network models (Cell 11):

\# Cell 11 --- Training curves for neural network models

FIGURES_DIR = Path("../outputs/figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

\# Plot train/val loss and accuracy for each neural model

\# Save as: outputs/figures/lab07_training_curves.png

- Compute and display the confusion matrix for all four models on the test set (Cell 12):

\# Cell 12 --- Confusion matrices for all four models

def get_predictions_dl(model, loader):

model.eval()

all_preds, all_labels = \[\], \[\]

with torch.no_grad():

for imgs, labels in loader:

out = model(imgs.to(DEVICE))

all_preds.extend(out.max(1)\[1\].cpu().numpy())

all_labels.extend(labels.numpy())

return np.array(all_labels), np.array(all_preds)

def get_predictions_ml(model, X_test):

return model.predict(X_test)

\# Generate a 1x4 or 2x2 grid of confusion matrices

\# Save as: outputs/figures/lab07_confusion_matrices.png

- Compute all performance metrics for all four models (Cell 13):

\# Cell 13 --- Metrics computation

def compute_metrics(y_true, y_pred, model_name, params=None, train_time=None):

acc = accuracy_score(y_true, y_pred)

prec = precision_score(y_true, y_pred, average='macro', zero_division=0)

rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f"\\n{'='\*50}")

print(f"Model: {model_name}")

print(f" Test Accuracy : {acc:.4f}")

print(f" Precision : {prec:.4f}")

print(f" Recall : {rec:.4f}")

print(f" F1-Score : {f1:.4f}")

if params: print(f" Parameters : {params:,}")

if train_time: print(f" Training Time : {train_time:.1f}s")

print(classification_report(y_true, y_pred,

target_names=CLASS_NAMES, zero_division=0))

return {'Model': model_name, 'Test Accuracy': f'{acc:.4f}',

'Precision': f'{prec:.4f}', 'Recall': f'{rec:.4f}',

'F1-Score': f'{f1:.4f}',

'Parameters': f'{params:,}' if params else 'N/A',

'Training Time (s)': f'{train_time:.1f}' if train_time else 'N/A'}

## **Part G) Generate Comparison Summary**

Create and save a consolidated comparison table for all four models (Cell 14). Fill in the table below after running your experiments:

| **Metric**        | **Model A** | **Model B** | **Model C** | **Model D** |
| ----------------- | ----------- | ----------- | ----------- | ----------- |
| Test Accuracy     |             |             |             |             |
| Precision (macro) |             |             |             |             |
| Recall (macro)    |             |             |             |             |
| F1-Score (macro)  |             |             |             |             |
| Training Time (s) |             |             |             |             |
| Parameters (#)    |             |             |             |             |
| Best Val Accuracy |             |             |             |             |
| Overfitting (Y/N) |             |             |             |             |

\# Cell 14 --- Summary comparison table

TABLES_DIR = Path("../outputs/tables")

TABLES_DIR.mkdir(parents=True, exist_ok=True)

rows = \[

compute_metrics(y_true_a, y_pred_a, MODEL_A_NAME, params_a, time_a),

compute_metrics(y_true_b, y_pred_b, MODEL_B_NAME, train_time=time_b),

compute_metrics(y_true_c, y_pred_c, MODEL_C_NAME, params_c, time_c),

compute_metrics(y_true_d, y_pred_d, MODEL_D_NAME, train_time=time_d),

\]

df = pd.DataFrame(rows)

print(df.to_string(index=False))

df.to_csv(TABLES_DIR / 'lab07_model_comparison.csv', index=False)

print("\\nSaved: lab07_model_comparison.csv")

- Generate a bar chart comparing test accuracy and F1-score across the four models. Save as: outputs/figures/lab07_accuracy_comparison.png (Cell 15)

\# Cell 15 --- Bar chart: Test Accuracy and F1-Score comparison

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

model_names = \[MODEL_A_NAME, MODEL_B_NAME, MODEL_C_NAME, MODEL_D_NAME\]

colors = \['#E74C3C', '#2980B9', '#27AE60', '#8E44AD'\]

for ax, (metric_key, ylabel) in zip(axes, \[

('Test Accuracy', 'Test Accuracy'),

('F1-Score', 'F1-Score (macro)')

\]):

vals = \[float(r\[metric_key\]) for r in rows\]

bars = ax.bar(model_names, vals, color=colors, edgecolor='white', width=0.5)

ax.set_ylim(0, 1)

ax.set_ylabel(ylabel)

ax.set_title(f'{ylabel} Comparison', fontweight='bold')

ax.bar_label(bars, fmt='%.3f', padding=3)

ax.tick_params(axis='x', rotation=20)

ax.grid(axis='y', alpha=0.3)

plt.suptitle('Four-Model Performance Comparison --- Lab 7',

fontsize=13, fontweight='bold')

plt.tight_layout()

plt.savefig(FIGURES_DIR / 'lab07_accuracy_comparison.png', dpi=150)

plt.show()

**RESULTS AND DISCUSSION**

## **A. Model Architecture Analysis**

For each of the four models, provide a written analysis addressing: (1) the architectural or algorithmic assumptions underlying the model and why they are appropriate (or limiting) for your dataset, (2) the role of each major component or hyperparameter in the model (e.g., kernel type, number of trees, convolutional depth), and (3) how the model's capacity relates to the size of your training data.

## **B. Training Behavior**

For all neural network models, analyze the training curves. Identify: convergence speed, presence of overfitting or underfitting, the epoch at which performance plateaus, and any instability in loss curves. For classical ML models, discuss the effect of key hyperparameters on cross-validation performance (if applicable).

## **C. Performance Evaluation**

Compare per-class precision, recall, and F1 across all four models using the confusion matrices and classification reports. Identify which classes are consistently misclassified and which models handle class imbalance best. Discuss any performance trade-offs observed (e.g., higher accuracy but lower recall on minority classes).

## **D. Computational Cost Analysis**

Compare training times and parameter counts across the four models. Which model offers the best accuracy-to-cost ratio? Consider practical scenarios: if deployment required real-time inference on a mobile device, which model would you choose and why?

## **E. Identified Limitations**

Discuss at least three limitations observed during your experiments. Consider: dataset size and class imbalance effects on model training, sensitivity of each model to hyperparameter choices, generalization gaps between validation and test performance, and misclassification patterns that persist across multiple models.

# **Questions (Answer Individually)**

- Why is it important to use the same dataset split (same random seed, same proportions) when comparing multiple models? What types of errors can arise if splits differ between models?
- Explain the No Free Lunch Theorem in your own words. How does this theorem motivate the multi-model comparison approach used in this laboratory?
- For imbalanced datasets, why is F1-score (macro-averaged) a more informative metric than raw accuracy? Provide an example using class distributions from your own dataset.
- Describe the key difference between a classical ML model (e.g., SVM or Random Forest) and a deep neural network in terms of feature learning. What is the role of feature engineering in each approach?
- Based on your experimental results, which of the four models would you recommend for deployment? Justify your choice using at least three specific pieces of evidence from your comparison table and confusion matrices.

Write a conclusion of 8-10 sentences in paragraph form summarizing:

**CONCLUSION**

- The rationale for selecting the four models and how they represent diverse algorithmic families.
- Key differences in training behavior, convergence, and stability observed across models.
- Which model achieved the best overall performance on your dataset and the specific metrics that support this conclusion.
- Observed trade-offs between model accuracy, computational cost, and complexity.
- Remaining challenges (e.g., persistent misclassification patterns, class imbalance issues) and planned improvements for future laboratories.