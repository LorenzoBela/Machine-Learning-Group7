MACHINE LEARNING AND PERCEPTION LAB**

Adamson University Computer Engineering Department

**Laboratory Exercise 6**

**Convolutional Neural Networks (CNNs) for Image-Based Perception**

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

- At the end of this laboratory exercise, the student should be able to:
- Explain the fundamental principles of CNNs and why they excel at image-based perception tasks.
- Design and compare three distinct CNN architectures of varying depth and complexity.
- Adapt a CNN training pipeline to a custom group-collected image dataset.
- Train, validate, and evaluate CNN models using loss curves, accuracy metrics, and confusion matrices.
- Apply data augmentation and regularization to improve generalization.
- Justify architectural choices by analyzing training behavior and test-set performance.

**DISCUSSION**

# **Introduction**

In previous laboratories, students explored fully connected networks (MLPs) applied to flattened image vectors. While effective for simple benchmarks, MLPs fail to exploit the spatial structure embedded in images - every pixel is treated independently, ignoring the rich neighborhood relationships that define edges, textures, and shapes.

Convolutional Neural Networks (CNNs) address this fundamental limitation. Inspired by the receptive-field organization of the mammalian visual cortex, CNNs use shared convolutional filters to detect local patterns anywhere in an image, regardless of position. This results in a dramatically more parameter-efficient and generalizable architecture.

This lab goes beyond a basic introduction. Each group will design, train, and compare THREE CNN architectures on their own collected image dataset, developing hands-on intuition for how depth, width, and regularization affect perception performance.

# **Detailed Discussion**

## **1\. Why Fully Connected Networks Are Insufficient for Images**

Feeding a 64×64 RGB image into an MLP requires a 12,288-dimensional input vector. The first hidden layer alone would have millions of weights - and the network still has no concept of spatial adjacency. CNNs solve this with three key inductive biases:

- Local connectivity - each neuron sees only a small patch (receptive field).
- Parameter sharing - one filter is reused across every position, drastically reducing parameters.
- Hierarchical composition - simple features (edges) combine into complex ones (objects) layer by layer.

## **2\. Core Principles of Convolutional Neural Networks**

### **a) The Convolution Operation**

A 2-D convolution slides a kernel K over input X to produce feature map Y:

Y(i,j) = Σ_m Σ_n X(i+m, j+n) · K(m,n)

Multiple kernels produce multiple feature maps, each detecting a different visual pattern. During training, kernel values are learned via backpropagation.

### **b) Activation Functions**

ReLU f(x) = max(0,x) is the standard activation in CNNs. It avoids the vanishing-gradient problem, is computationally cheap, and allows sparse activations that improve generalization.

### **c) Pooling Layers**

Max-pooling selects the largest value in each local window (e.g., 2×2), halving spatial dimensions. Benefits include reduced computation, translation invariance, and implicit regularization.

## **3\. Batch Normalization and Dropout**

Batch Normalization normalizes layer activations to zero mean and unit variance within each mini-batch. This accelerates convergence, allows higher learning rates, and acts as a mild regularizer.

Dropout randomly zeroes a fraction of neuron outputs during training (e.g., p = 0.4-0.5), acting as an ensemble of sub-networks and preventing co-adaptation of features.

## **4\. The Three CNN Architectures Used in This Lab**

Each group will implement, train, and compare the following three models on their custom dataset:

| **Model** | **Nickname**   | **Depth**       | **Key Feature**                              |
| --------- | -------------- | --------------- | -------------------------------------------- |
| Model A   | ShallowCNN     | 2 conv blocks   | Baseline - minimal depth, no regularization  |
| Model B   | RegularizedCNN | 3 conv blocks   | Adds BatchNorm + Dropout for stability       |
| Model C   | DeepCNN        | 4 stages + skip | Residual connection + Global Average Pooling |

**MATERIALS**

## **Hardware**

- Laptop/PC with at least 8 GB RAM (GPU recommended for faster training)

## **Software**

- Python 3.10+
- Jupyter Notebook / Google Colab

## **Libraries**

- torch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- pandas
- PIL (Pillow)

## **Dataset / Data Source**

- Group-collected image dataset (organized per class in subfolders)

**PROCEDURES**

## **Part A) Environment and Dataset Setup**

- Create a directory: ml-perception-labs/lab06_cnn/
- Create the following folder structure if not already present:

![](data:image/png;base64,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)

- Create a notebook named: notebook/Lab06_CNN_ImagePerception.ipynb
- In the first notebook cell, display: Name, Section, Date, Dataset name
- Install and import all required libraries (Cell 1 - see sample code below):

\# Cell 1 - Install and import dependencies

!pip install torch torchvision matplotlib scikit-learn seaborn pandas pillow

import torch, torchvision, numpy as np, matplotlib.pyplot as plt

import seaborn as sns, pandas as pd

from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report

print(f"PyTorch : {torch.\__version_\_}")

print(f"CUDA : {torch.cuda.is_available()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")

## **Part B) Load and Prepare Dataset**

Organize your collected images so each class is a subfolder under data/raw/. The ImageFolder loader will automatically assign labels from folder names.

- Arrange your dataset in the following structure:

data/raw/

├── class_0/ (e.g., "cat")

│ ├── img001.jpg

│ └── ...

├── class_1/ (e.g., "dog")

│ └── ...

└── class_N/

- Define transforms - apply augmentation only during training (Cell 2):

\# Cell 2 - Transforms

import torchvision.transforms as transforms

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

- Load dataset and split 70% train / 15% val / 15% test (Cell 3):

\# Cell 3 - Load and split dataset

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader, random_split

DATA_DIR = Path("../data/raw")

full_dataset = ImageFolder(root=DATA_DIR, transform=train_transform)

CLASS_NAMES = full_dataset.classes

NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

print(f"Total images: {len(full_dataset)}")

n_total = len(full_dataset)

n_train = int(0.70 \* n_total)

n_val = int(0.15 \* n_total)

n_test = n_total - n_train - n_val

train_set, val_set, test_set = random_split(

full_dataset, \[n_train, n_val, n_test\],

generator=torch.Generator().manual_seed(SEED)

)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_set)} Val: {len(val_set)} Test: {len(test_set)}")

- Display example images per class and save to outputs/figures/lab06_dataset_samples.png

## **Part C) Define CNN Architectures**

Implement all three CNN models in separate cells. Document all design decisions in comments.

### **Model A - ShallowCNN (Baseline)**

A minimal 2-block CNN with no regularization. Serves as the performance floor.

\# Cell 4 - Model A: ShallowCNN

import torch.nn as nn

import torch.nn.functional as F

class ShallowCNN(nn.Module):

"""Baseline: 2 conv blocks, no regularization."""

def \__init_\_(self, num_classes):

super().\__init_\_()

self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

self.pool = nn.MaxPool2d(2, 2)

flat_size = 64 \* (IMG_SIZE // 4) \* (IMG_SIZE // 4)

self.fc1 = nn.Linear(flat_size, 256)

self.fc2 = nn.Linear(256, num_classes)

def forward(self, x):

x = self.pool(F.relu(self.conv1(x))) # Block 1

x = self.pool(F.relu(self.conv2(x))) # Block 2

x = torch.flatten(x, 1)

x = F.relu(self.fc1(x))

return self.fc2(x)

model_a = ShallowCNN(NUM_CLASSES).to(DEVICE)

params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)

print(f"ShallowCNN | Parameters: {params_a:,}")

### **Model B - RegularizedCNN (Intermediate)**

Adds a third convolutional block, BatchNorm after each convolution, and Dropout in the classifier head.

\# Cell 5 - Model B: RegularizedCNN

class RegularizedCNN(nn.Module):

"""3-block CNN with BatchNorm + Dropout for reduced overfitting."""

def \__init_\_(self, num_classes, dropout_p=0.4):

super().\__init_\_()

self.block1 = nn.Sequential(

nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),

nn.ReLU(inplace=True), nn.MaxPool2d(2,2),

)

self.block2 = nn.Sequential(

nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),

nn.ReLU(inplace=True), nn.MaxPool2d(2,2),

)

self.block3 = nn.Sequential(

nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),

nn.ReLU(inplace=True), nn.MaxPool2d(2,2),

)

flat_size = 128 \* (IMG_SIZE // 8) \* (IMG_SIZE // 8)

self.classifier = nn.Sequential(

nn.Dropout(p=dropout_p),

nn.Linear(flat_size, 512), nn.ReLU(inplace=True),

nn.Dropout(p=dropout_p),

nn.Linear(512, num_classes),

)

def forward(self, x):

x = self.block3(self.block2(self.block1(x)))

return self.classifier(torch.flatten(x, 1))

model_b = RegularizedCNN(NUM_CLASSES).to(DEVICE)

params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)

print(f"RegularizedCNN | Parameters: {params_b:,}")

### **Model C - DeepCNN with Skip Connection (Advanced)**

A 4-stage network with a residual block and Global Average Pooling. Residual (skip) connections pass gradients directly through layers, addressing the vanishing gradient problem in deeper networks.

\# Cell 6 - Model C: DeepCNN with residual block

class ResidualBlock(nn.Module):

"""Two conv layers + identity shortcut (1x1 conv if channel dims differ)."""

def \__init_\_(self, in_ch, out_ch):

super().\__init_\_()

self.conv_path = nn.Sequential(

nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),

nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),

nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),

nn.BatchNorm2d(out_ch),

)

self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, bias=False)

if in_ch != out_ch else nn.Identity())

self.relu = nn.ReLU(inplace=True)

def forward(self, x):

return self.relu(self.conv_path(x) + self.shortcut(x))

class DeepCNN(nn.Module):

"""4-stage deep CNN with skip connection and Global Average Pooling."""

def \__init_\_(self, num_classes, dropout_p=0.5):

super().\__init_\_()

self.stem = nn.Sequential(

nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32),

nn.ReLU(inplace=True), nn.MaxPool2d(2,2),

)

self.stage1 = nn.Sequential(

nn.Conv2d(32,64,3,padding=1,bias=False), nn.BatchNorm2d(64),

nn.ReLU(inplace=True), nn.MaxPool2d(2,2),

)

self.stage2 = ResidualBlock(64, 128)

self.stage3 = nn.Sequential(nn.MaxPool2d(2,2), ResidualBlock(128, 256))

self.gap = nn.AdaptiveAvgPool2d(1) # Global Average Pool → (B,256,1,1)

self.classifier = nn.Sequential(

nn.Dropout(p=dropout_p),

nn.Linear(256, 128), nn.ReLU(inplace=True),

nn.Linear(128, num_classes),

)

def forward(self, x):

x = self.stage3(self.stage2(self.stage1(self.stem(x))))

return self.classifier(torch.flatten(self.gap(x), 1))

model_c = DeepCNN(NUM_CLASSES).to(DEVICE)

params_c = sum(p.numel() for p in model_c.parameters() if p.requires_grad)

print(f"DeepCNN | Parameters: {params_c:,}")

## **Part D) Define Loss Function and Optimizer**

Use Cross-Entropy Loss for multi-class classification and Adam optimizer with StepLR scheduler. Justify all hyperparameter choices in code comments.

\# Cell 7 - Loss, optimizer, scheduler factory functions

import torch.optim as optim

NUM_EPOCHS = 30

LR = 1e-3

def make_optimizer(model):

\# Adam: adaptive learning rates per parameter; weight_decay = L2 regularization

return optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

def make_scheduler(optimizer):

\# Halve LR every 10 epochs to fine-tune in later training stages

return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

criterion = nn.CrossEntropyLoss() # = LogSoftmax + NLLLoss

print("Loss : CrossEntropyLoss")

print("Optimizer : Adam (lr=1e-3, weight_decay=1e-4)")

print("Scheduler : StepLR (step=10, gamma=0.5)")

## **Part E) Train All Three CNN Models**

Use the shared training loop below for all three models. The loop saves the best weights (by validation accuracy) and restores them after training.

\# Cell 8 - Reusable training loop

def train_model(model, train_loader, val_loader, num_epochs, model_name):

optimizer = make_optimizer(model)

scheduler = make_scheduler(optimizer)

history = {'train_loss':\[\], 'val_loss':\[\], 'train_acc':\[\], 'val_acc':\[\]}

best_val_acc, best_weights = 0.0, None

for epoch in range(1, num_epochs + 1):

\# -- Training --

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

\# -- Validation --

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

best_weights = {k: v.clone() for k,v in model.state_dict().items()}

if epoch % 5 == 0 or epoch == 1:

print(f"\[{model_name}\] Ep {epoch:3d} | "

f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} | "

f"ValLoss={val_loss:.4f} Acc={val_acc:.4f}")

model.load_state_dict(best_weights)

print(f" Best Val Acc ({model_name}): {best_val_acc:.4f}")

return history

\# Cell 9 - Train all three models

print("="\*55)

print("Training Model A - ShallowCNN")

history_a = train_model(model_a, train_loader, val_loader, NUM_EPOCHS, "ShallowCNN")

print("\\n" + "="\*55)

print("Training Model B - RegularizedCNN")

history_b = train_model(model_b, train_loader, val_loader, NUM_EPOCHS, "RegularizedCNN")

print("\\n" + "="\*55)

print("Training Model C - DeepCNN")

history_c = train_model(model_c, train_loader, val_loader, NUM_EPOCHS, "DeepCNN")

Record training loss, validation loss, training accuracy, and validation accuracy per epoch for all three models.

## **Part F) Evaluate CNN Performance**

Plot training curves and confusion matrices for all three models. Save all figures to the outputs/figures/ directory.

\# Cell 10 - Training curves (loss and accuracy)

FIGURES_DIR = Path("../outputs/figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

info = \[(history_a,'ShallowCNN','#E74C3C'),

(history_b,'RegularizedCNN','#2980B9'),

(history_c,'DeepCNN','#27AE60')\]

epochs = range(1, NUM_EPOCHS + 1)

for col, (hist, name, color) in enumerate(info):

for row, (key_tr, key_vl, ylabel) in enumerate(\[

('train_loss','val_loss','Loss'),

('train_acc','val_acc','Accuracy')

\]):

ax = axes\[row, col\]

ax.plot(epochs, hist\[key_tr\], label='Train', color=color, lw=2)

ax.plot(epochs, hist\[key_vl\], label='Val', color=color, lw=2, ls='--')

ax.set_title(f'{name}\\n{ylabel}', fontweight='bold')

ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)

if ylabel=='Accuracy': ax.set_ylim(0,1)

ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Training Curves - All Three CNN Models', fontsize=14, fontweight='bold')

plt.tight_layout()

plt.savefig(FIGURES_DIR / "lab06_training_curves.png", dpi=150)

plt.show()

\# Cell 11 - Confusion matrices

def get_predictions(model, loader):

model.eval()

all_preds, all_labels = \[\], \[\]

with torch.no_grad():

for imgs, labels in loader:

out = model(imgs.to(DEVICE))

all_preds.extend(out.max(1)\[1\].cpu().numpy())

all_labels.extend(labels.numpy())

return np.array(all_labels), np.array(all_preds)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (mdl, name, cmap) in zip(axes, \[

(model_a,'ShallowCNN','Blues'),

(model_b,'RegularizedCNN','Greens'),

(model_c,'DeepCNN','Oranges')

\]):

y_true, y_pred = get_predictions(mdl, test_loader)

cm = confusion_matrix(y_true, y_pred).astype(float)

cm /= cm.sum(axis=1, keepdims=True)

sns.heatmap(cm, annot=True, fmt='.2f', ax=ax,

xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap=cmap)

acc = (y_true==y_pred).mean()

ax.set_title(f'{name}\\nTest Acc: {acc:.4f}', fontweight='bold')

ax.set_xlabel('Predicted'); ax.set_ylabel('True')

ax.tick_params(axis='x', rotation=45)

plt.suptitle('Normalized Confusion Matrices', fontsize=13, fontweight='bold')

plt.tight_layout()

plt.savefig(FIGURES_DIR / "lab06_confusion_matrix.png", dpi=150)

plt.show()

## **Part G) Compare CNN Models and Summarize Results**

Generate and save a summary comparison table for all three models.

\# Cell 12 - Summary comparison table

import pandas as pd

TABLES_DIR = Path("../outputs/tables")

TABLES_DIR.mkdir(parents=True, exist_ok=True)

rows = \[\]

for mdl, name, hist in \[(model_a,'ShallowCNN',history_a),

(model_b,'RegularizedCNN',history_b),

(model_c,'DeepCNN',history_c)\]:

y_true, y_pred = get_predictions(mdl, test_loader)

acc = (y_true==y_pred).mean()

params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

best_val = max(hist\['val_acc'\])

rows.append({'Model': name,

'Parameters': f'{params:,}',

'Best Val Acc': f'{best_val:.4f}',

'Test Accuracy': f'{acc:.4f}'})

df = pd.DataFrame(rows)

print(df.to_string(index=False))

df.to_csv(TABLES_DIR / "lab06_model_comparison.csv", index=False)

print("\\nSaved: lab06_model_comparison.csv")

**RESULTS AND DISCUSSION**

## **A. CNN Architecture Analysis**

For each of the three models, describe: (1) each layer and its role in feature extraction, (2) rationale for the number of filters chosen, (3) how the receptive field grows with depth, and (4) the expected impact of BatchNorm, Dropout, and skip connections.

## **B. Training Behavior**

Analyze the loss and accuracy curves for each model. Identify: convergence speed, presence of overfitting (training accuracy significantly higher than validation accuracy), the epoch at which overfitting begins (if any), and how regularization techniques in Models B and C change behavior compared to Model A.

## **C. Performance Evaluation**

Discuss per-class accuracy from the confusion matrices. Which classes are hardest to classify and why? Are there any confusion pairs that make intuitive sense (visually similar classes)?

## **E. Identified Limitations**

Discuss at least three limitations observed during your experiments. Consider: dataset size and class imbalance, computational cost differences between models, sensitivity to hyperparameters, and remaining misclassification patterns that even the best model could not resolve.

# **Questions (Answer Individually)**

**1\.** Why are CNNs fundamentally better suited for image data than MLPs? Use the concepts of spatial locality and parameter sharing in your answer.

**2\.** How does the receptive field of a CNN neuron increase with network depth? How does this relate to hierarchical feature learning?

**3\.** What specific problem does Batch Normalization solve, and how did it affect Model B's training curve compared to Model A in your experiment?

**4\.** What is a residual (skip) connection as used in Model C, and what gradient-related problem does it address in deep networks?

**5\.** How does Global Average Pooling (used in Model C) differ from a traditional Flatten + Dense head? What are the advantages for your dataset?

**CONCLUSION**

Write a conclusion including:

- Write a conclusion of 8-10 sentences in paragraph form summarizing:
- Key architectural differences between MLPs and CNNs observed in this laboratory.
- The specific roles of convolutional layers, pooling, BatchNorm, Dropout, and skip connections.
- Which of the three CNN models performed best on your group's dataset and why.
- Observed overfitting/underfitting patterns and how regularization addressed them.
- Remaining challenges and planned improvements for Lab 8 (hyperparameter tuning).