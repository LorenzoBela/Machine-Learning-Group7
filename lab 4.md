Adamson University Computer Engineering Department

**Laboratory Exercise 4  
Machine Learning for Perception Lab 4: Exploratory Data Analysis (EDA) and Dataset Bias Assessment**

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

- Perform exploratory data analysis (EDA) on perception datasets using quantitative and visual techniques.
- Analyze class distributions, feature characteristics, and data variability in image datasets.
- Identify potential sources of bias and imbalance in perception datasets.
- Interpret EDA results to guide model design, augmentation, and evaluation decisions.
- Document dataset limitations and risks that may affect model fairness and generalization.

**DISCUSSION**

Introduction

Before training any machine learning model, especially deep learning models for perception, it is essential to understand the data itself. Exploratory Data Analysis (EDA) is the systematic process of examining datasets to uncover patterns, anomalies, biases, and limitations.

In perception systems, poor dataset understanding can lead to:

- Overconfident models with misleading accuracy
- Bias toward specific classes or visual patterns
- Poor generalization to real-world conditions

This laboratory focuses on applying EDA techniques to image-based perception datasets, allowing students to make **informed design decisions** before model training begins.

**1\. What is Exploratory Data Analysis (EDA)?**

EDA is the process of **summarizing, visualizing, and interpreting data** before applying formal modeling techniques. Unlike model training, EDA does not attempt to predict outcomes; instead, it aims to answer questions such as:

- What does the data look like?
- How are classes distributed?
- Are there patterns or anomalies?
- What assumptions can we safely make?

EDA is a **diagnostic step** that helps avoid incorrect modeling assumptions.

**2\. Why EDA Is Critical in Perception-Based ML**

Perception datasets differ from tabular data because:

- Features are high-dimensional (pixels)
- Relationships are spatial, not explicit
- Visual similarity may not align with class labels

Without EDA:

- A model may learn **background artifacts**
- Minority classes may be ignored
- Evaluation metrics may hide failure cases

**Key Insight:**

You cannot trust a model more than you understand the data it was trained on.

**3\. Types of EDA for Image Data**

**a) Structural Analysis**

Focuses on dataset properties:

- Number of samples
- Image size and channels
- Data type and value ranges

Structural inconsistencies can cause:

- Training errors
- Inconsistent batch processing
- Implicit bias in preprocessing

**b) Distribution Analysis**

Examines how data is distributed:

- Class counts
- Sample frequency per class
- Balance vs imbalance

Even "balanced" datasets can still have:

- Intra-class imbalance
- Feature dominance within classes

**c) Visual Pattern Analysis**

Uses image visualization to:

- Inspect background dominance
- Identify confusing classes
- Observe viewpoint, lighting, and scale variations

This step is crucial for perception tasks, where visual context heavily influences learning.

**d) Feature-Level Analysis (Pixel Statistics)**

Although images are high-dimensional, basic statistics still matter:

- Mean and standard deviation of pixel values
- Channel-wise distributions
- Contrast and brightness variability

These insights inform:

- Normalization choices
- Augmentation parameters
- Model architecture decisions

**4\. Dataset Bias in Perception Systems**

Dataset bias occurs when a dataset **does not represent the real-world population or conditions** under which the model will operate.

**Common Bias Sources:**

- Class imbalance
- Background bias (object always appears in same context)
- Acquisition bias (lighting, camera quality)
- Selection bias (easy cases overrepresented)

Bias does not always mean ethical bias-it can also mean **technical bias** that reduces generalization.

**5\. Relationship Between EDA and Model Performance**

EDA directly influences:

- Choice of augmentation techniques
- Selection of loss functions
- Metric interpretation
- Error analysis strategies

Skipping EDA often results in:

- High training accuracy
- Unexpected test failures
- Misleading evaluation metrics

**6\. EDA as a Tool for Responsible AI**

EDA supports responsible AI by:

- Exposing dataset limitations early
- Encouraging transparency and documentation
- Preventing misuse of models beyond dataset scope

In academic and industrial settings, EDA findings are often included in **dataset cards, model cards, and validation reports**.

**MATERIALS**

**Hardware**

- Laptop/PC with at least 8GB RAM (recommended)

**Software**

- Python 3.10+
- Jupyter Notebook / Google Colab

**Libraries**

- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- torchvision
- PIL (Pillow)

**Dataset / Data Source**

- Collected Data Set

**PROCEDURES**

**Part A)** Environment and Dataset Setup

- Create a directory:

_ml-perception-labs/lab04_eda_bias/_

- ![](data:image/png;base64,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)Create the following structure if not already present:
- Create a notebook named:

_notebook/Lab04_EDA_BiasAssessment.ipynb_

- In the first notebook cell, display:

- Name
- Section
- Date
- Dataset name

**Part B)** Load Dataset

- Load your Data from data/raw/.
- Display:

- Total samples
- Class names
- Image shape and data type

**Part C)** Structural Dataset Analysis

- Verify image dimensions for a subset of samples.
- Check:

- Number of channels
- Pixel value range

- Report any inconsistencies.

Save findings in:

outputs/logs/lab04_structure_check.txt

**Part C)** Class Distribution Analysis

- Count samples per class.
- Create:

- Table of class counts
- Bar chart of class distribution

Save:

outputs/tables/lab04_class_distribution.csv  
outputs/figures/lab04_class_distribution.png

**Part E)** Visual Pattern Inspection

- For each class:
  - Display 5 random images
- Observe:
  - Background similarity
  - Object position
  - Scale and orientation

Save a grid figure:

outputs/figures/lab04_class_samples.png

**Part F)** Pixel-Level Statistical Analysis

- Compute:

- Mean pixel value per channel
- Standard deviation per channel

- Visualize channel-wise distributions (histograms).

Save:

outputs/figures/lab04_pixel_histograms.png

**Part G)** Bias Identification and Interpretation

- Identify at least two potential biases.
- For each bias:

- Describe the cause
- Explain its impact on model learning
- Suggest one mitigation strategy

Save written analysis in:

outputs/logs/lab04_bias_analysis.txt

**RESULTS AND DISCUSSION**

A. Dataset Overview

Summarize dataset size, structure, and general observations.

B. Distribution Findings

Discuss whether the dataset is balanced and why that matters.

C. Visual Pattern Observations

Describe visual similarities/differences across classes.

D. Pixel Statistics Interpretation

Explain how pixel distributions affect preprocessing and training.

E. Bias Analysis

Explain identified biases and proposed mitigations.

**Questions (Answer Individually)**

What is the purpose of EDA in machine learning for perception?
EDA helps us understand the dataset before training a model. It shows class balance, image quality, and possible problems like noise or missing patterns. It also tells us if data is enough for each class. With this, we can choose better preprocessing and model settings.

Why is visual inspection necessary even when datasets are numerically balanced?
Numbers can look balanced, but images may still have hidden issues. Some classes can share similar backgrounds, lighting, or camera angles that can bias learning. Some images may also be blurry or badly cropped. Visual checking helps us catch these problems early.

What is one example of technical bias in perception datasets?
One example is background bias, where one class often appears with the same background. The model may learn the background instead of the object itself. This can reduce accuracy on real-world images. It becomes worse when test images have different scenes.

How can EDA results influence augmentation strategies?
EDA shows what variations are missing, like low lighting, rotations, or scale changes. Based on that, we choose augmentations that make training data more realistic and diverse. We avoid random augmentations that do not match dataset needs. This improves generalization.

Why should EDA be conducted before model training?
EDA is done first so we can fix data problems before they affect learning. It helps avoid wasted training time on poor or biased data. It also leads to fairer and more reliable model results. In short, better data understanding gives better model performance.

**CONCLUSION**

This EDA helped us clearly understand the dataset before training, including class balance, image quality, and pixel behavior. We found important limitations such as possible background bias, uneven visual conditions, and some samples that may not represent real-world scenes. These findings show why bias awareness is important, because hidden bias can make a model look good in testing but fail in actual use. Overall, the EDA results guide the next stage by helping us choose better preprocessing, targeted augmentation, and fairer evaluation steps for CNN training.