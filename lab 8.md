MACHINE LEARNING AND PERCEPTION LAB**

Adamson University Computer Engineering Department

**Laboratory Exercise 8**

**Fine-Tuning of the Two Best-Performing Models**

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

- Identify and justify the two best-performing models from Laboratory Exercise 7 based on empirical evidence and computational cost.
- Apply systematic fine-tuning techniques (e.g., learning-rate scheduling, layer freezing, hyperparameter search, regularization, and data augmentation) to improve each selected model on the group's dataset.
- Adapt the fine-tuning workflow to the appropriate task type - image classification, object detection, or semantic / instance segmentation - depending on each chosen model.
- Quantitatively compare baseline (pre-fine-tuning) and tuned (post-fine-tuning) performance using task-appropriate metrics.
- Recommend a single deployment-ready model based on the fine-tuning outcomes, accuracy gains, and resource trade-offs.

**DISCUSSION**

# **Introduction**

In Laboratory Exercise 7, students retrained four candidate machine learning or deep learning models on their group dataset and compared their performance using consistent preprocessing, an identical data split, and a common evaluation protocol. The exercise produced empirical evidence about which model families are best suited to the dataset and which are not.

Laboratory Exercise 8 advances this study from comparative selection to focused refinement. Rather than introducing additional model families, this lab takes the two best-performing models from Lab 7 and improves each through fine-tuning. Fine-tuning is the practice of adjusting a model's parameters, hyperparameters, or training procedure to better fit a target dataset and task without redesigning the underlying architecture. It is the standard step that converts a strong baseline into a deployment-ready system.

Because each group may have selected different model types in Lab 7 including classification models, object detectors, or segmentation networks this laboratory describes the fine-tuning workflow at the conceptual and procedural level. Each group is responsible for adapting the steps to the specific task and framework appropriate to its two chosen models.

# **Detailed Discussion**

## **1\. From Model Selection to Model Refinement**

Model selection answers the question, "Which algorithm family is most promising for this problem?" Fine-tuning answers the follow-up question, "How do we extract the maximum performance from the chosen algorithm given the data, the task, and the available compute?"

Fine-tuning is rarely a single action. It is an iterative loop that involves: (1) diagnosing the limitations of the baseline (overfitting, underfitting, bias toward majority classes, poor localization, weak boundary detection, etc.), (2) hypothesizing a modification, (3) retraining only what is necessary, and (4) re-evaluating against the same held-out test set used during model selection.

## **2\. Core Fine-Tuning Concepts**

**a) Layer Freezing and Selective Updating.** Pretrained networks contain low-level feature extractors (edges, textures, simple shapes) that often transfer well to new datasets. Freezing early layers and only updating the later, task-specific layers reduces the risk of catastrophic forgetting and lowers training cost. Progressive unfreezing unlocking deeper layers gradually is a common strategy when the dataset is large enough to support full updates.

**b) Learning-Rate Strategy.** Fine-tuning typically requires a smaller learning rate than training from scratch, because the model is already close to a good solution. Differential or layer-wise learning rates (smaller for early layers, larger for newly initialized heads) often outperform a single global rate. Schedules such as cosine annealing, step decay, or warm restarts further stabilize convergence.

**c) Regularization Adjustments.** If the baseline overfits, fine-tuning may introduce or strengthen dropout, weight decay, label smoothing, or early stopping. If the baseline underfits, regularization is reduced or removed and capacity is increased instead.

**d) Data Augmentation Tuning.** Augmentation is task-dependent: random cropping and flipping are safe for classification, but segmentation and detection require synchronized transformations of both images and annotations (masks or bounding boxes). Stronger augmentation (e.g., MixUp, CutMix, Mosaic) is appropriate when overfitting persists.

**e) Loss-Function Refinement.** Class imbalance, hard examples, and small objects often benefit from specialized losses focal loss for imbalanced classification or detection, Dice or Tversky loss for segmentation, and IoU-based regression losses (GIoU, CIoU, DIoU) for bounding-box prediction.

**f) Optimizer and Batch-Size Choices.** Switching from SGD to Adam, AdamW, or vice versa can change convergence behavior. Larger batch sizes stabilize gradients but require linear-scaling or warmup adjustments to the learning rate.

## **3\. Hyperparameter Tuning Strategies**

Fine-tuning is a search problem. The candidate strategies, in order of increasing rigor, are:

- Manual tuning - adjusting one hyperparameter at a time based on diagnostic plots (loss curves, confusion matrices, error maps). Effective for small experiments and quick iterations.
- Grid search - exhaustively evaluating combinations from a predefined grid. Reliable but expensive.
- Random search - sampling combinations randomly within defined ranges. Often more efficient than grid search for high-dimensional spaces.
- Bayesian / sequential optimization (e.g., Optuna, Ray Tune, Hyperopt) - using the results of past trials to focus the search. Recommended when each training run is costly.

Regardless of method, every group must record each trial: the configuration tested, the validation metric obtained, and a brief note on the outcome. This audit trail is part of the deliverable.

## **4\. Fine-Tuning Across Different Task Types**

Each group's two best models may target different computer-vision tasks. The fine-tuning principles above apply universally, but the specific levers differ by task type.

**a) Image Classification**

- Replace or re-initialize the final classification head to match the number of classes.
- Use class-weighted loss or oversampling when the dataset is imbalanced.
- Diagnose errors using the confusion matrix and per-class precision / recall.
- Apply test-time augmentation (TTA) only after baseline tuning has plateaued.

**b) Object Detection**

- Adjust anchor boxes, anchor scales, or anchor-free configurations to match the object sizes in the dataset.
- Tune the IoU threshold for positive/negative anchor matching and the Non-Maximum Suppression (NMS) threshold for inference.
- Balance the classification and bounding-box regression loss weights; small objects often require higher regression weight or a dedicated small-object branch.
- Diagnose errors using per-class Average Precision (AP), precision-recall curves, and visual inspection of false positives and missed detections.

**c) Semantic / Instance Segmentation**

- Fine-tune the encoder (often a pretrained backbone) at a lower learning rate while training the decoder at a higher rate.
- Use a combined loss (e.g., Cross-Entropy + Dice) to balance pixel-level accuracy with region-level overlap.
- Address class imbalance through weighted loss, focal loss, or boundary-aware loss for thin or rare structures.
- Diagnose errors using per-class IoU, qualitative overlay maps, and boundary-IoU metrics for fine structures.

## **5\. Performance Metrics by Task Type**

The two fine-tuned models must be evaluated using metrics appropriate to their respective tasks. Reporting accuracy alone is insufficient if either model is a detector or a segmentation network. The table below summarizes the recommended metrics and loss functions.

| **Task Type**             | **Primary Metrics**                                                                  | **Common Loss Functions**                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **Image Classification**  | Accuracy, Precision, Recall, F1-Score (macro), Top-k Accuracy, Confusion Matrix      | Cross-Entropy Loss, Focal Loss, Label-Smoothing Loss                                         |
| **Object Detection**      | mAP@0.5, mAP@\[0.5:0.95\], Per-class AP, Precision-Recall Curve, IoU, Inference FPS  | Classification + Bounding-Box Regression Loss (e.g., Smooth L1, GIoU, CIoU), Objectness Loss |
| **Semantic Segmentation** | mean Intersection over Union (mIoU), Pixel Accuracy, Dice Coefficient, Per-class IoU | Cross-Entropy (per-pixel), Dice Loss, Tversky Loss, Combined CE + Dice                       |
| **Instance Segmentation** | Mask AP, Mask AP@0.5, Boundary IoU, Detection AP                                     | Mask Loss + Detection Losses (combined multi-task loss)                                      |

## **6\. Avoiding Common Pitfalls**

- Do not change the test set between baseline (Lab 7) and fine-tuned (Lab 8) evaluations. Both models must be tested on the identical held-out test split with the same random seed.
- Avoid "tuning on the test set." All hyperparameter decisions must be based on validation-set performance only.
- Track every experimental change. An untracked tweak that yields a small gain is indistinguishable from random variation.
- Beware of false improvements caused by data leakage - e.g., applying augmentations on validation images, or letting the same image appear in both train and test splits.

**MATERIALS**

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

- Same group-collected image dataset used in Laboratory Exercise 6 and 7
- Dataset must be organized per class in subfolders (ImageFolder format)

**PROCEDURES**

## **Part A) Project Setup**

- Create a directory: ml-perception-labs/lab08_finetuning/
- Replicate the Lab 7 folder structure, adding a /tuning_logs/ subdirectory for hyperparameter search records and a /finetuned_models/ subdirectory for refined model weights:

lab08_finetuning/

├── data/

│ └── raw/ same dataset from Lab 6 / Lab 7

├── notebook/

│ └── Lab08_FineTuning.ipynb

├── finetuned_models/ saved fine-tuned weights

├── tuning_logs/ hyperparameter search records (CSV / JSON)

└── outputs/

├── figures/

└── tables/

- Create a notebook named notebook/Lab08_FineTuning.ipynb.
- In the first notebook cell, display: Name, Section, Date, Dataset name, the two selected models from Lab 7, and the task type targeted by each (classification / object detection / segmentation).

## **Part B) Selection of the Two Best Models from Lab 7**

Review the consolidated comparison table from Lab 7. Select the two best-performing models using the following decision rule:

- Primary criterion - highest task-appropriate metric (test accuracy for classification, mAP for detection, mIoU for segmentation) on the held-out test set.
- Tie-breaker 1 - generalization gap: smaller difference between training and validation performance.
- Tie-breaker 2 - computational cost: shorter training time and fewer parameters at comparable accuracy.

Complete the table below before proceeding. The justification column must explain why each model was the strongest candidate from Lab 7.

| **Rank**  | **Model Architecture / Name** | **Task Type** | **Lab 7 Performance** | **Justification for Inclusion** |
| --------- | ----------------------------- | ------------- | --------------------- | ------------------------------- |
| **Top 1** |                               |               |                       |                                 |
| **Top 2** |                               |               |                       |                                 |

## **Part C) Diagnosing the Baseline**

Before tuning anything, characterize what is wrong with each baseline. A well-defined diagnosis directs the fine-tuning effort and prevents random tweaking. For each of the two selected models, document the following:

- Inspect the Lab 7 training and validation curves. Mark the epoch at which the model began to overfit, plateaued, or diverged.
- Re-examine the Lab 7 confusion matrix (classification), per-class AP table (detection), or per-class IoU table (segmentation). Identify the classes or categories with the weakest performance.
- Visually inspect a sample of failure cases from the test set. Save at least six failure examples per model to outputs/figures/.
- State a written hypothesis describing the dominant failure mode (e.g., "the detector misses small objects," "the classifier confuses two visually similar classes," "the segmentation model produces fragmented masks at object boundaries").

## **Part D) Fine-Tuning Plan**

Based on the diagnosis, write a fine-tuning plan for each model before running any experiment. Each plan must include the proposed change, the expected effect, and the metric used to verify the effect. Fill in the table below.

| **Hyperparameter / Setting**                     | **Top Model #1** | **Top Model #2** |
| ------------------------------------------------ | ---------------- | ---------------- |
| **Baseline value (from Lab 7)**                  |                  |                  |
| **New tuned value**                              |                  |                  |
| **Learning rate (and schedule)**                 |                  |                  |
| **Optimizer**                                    |                  |                  |
| **Batch size**                                   |                  |                  |
| **Number of epochs / iterations**                |                  |                  |
| **Layer-freezing strategy**                      |                  |                  |
| **Data augmentation strategy**                   |                  |                  |
| **Regularization (dropout, weight decay, etc.)** |                  |                  |
| **Task-specific loss function**                  |                  |                  |
| **Early-stopping criterion**                     |                  |                  |

Each plan must address at least three of the following levers: learning-rate schedule, layer-freezing strategy, regularization (dropout / weight decay / label smoothing), data augmentation pipeline, optimizer choice, task-specific loss function, batch size, or input resolution. A plan that changes only one hyperparameter is insufficient.

## **Part E) Execute Fine-Tuning**

Run the fine-tuning experiments for each of the two models. The implementation will depend on the framework in use (PyTorch, Ultralytics, Detectron2, MMDetection, MMSegmentation, scikit-learn, etc.) and the task type. Each group is responsible for writing the fine-tuning code appropriate to its models. The following requirements apply to all groups regardless of framework:

- Use the same random seed (SEED = 42) and the same train / val / test split as in Labs 6 and 7.
- Begin from the saved Lab 7 weights of each selected model (do not retrain from scratch unless the model is a classical ML model that does not support warm-starting).
- Run a minimum of three fine-tuning trials per model with different hyperparameter configurations drawn from the plan in Part D. Record each trial in tuning_logs/ with: trial ID, configuration, validation metric, training time, and a brief note.
- Select the best trial per model based on validation-set performance only. Save its weights to finetuned_models/.
- Log training-time, GPU memory usage (if applicable), and per-epoch metrics for the selected best trial of each model.

### **Google Colab Optimization (if training on Colab)**

- Set the Colab runtime to GPU and confirm GPU availability before starting any trial.
- Copy the dataset to the local Colab runtime for training speed. Keep Drive for persistence and sync outputs and logs back at the end of each trial.
- Use mixed precision when supported to reduce memory use and improve throughput.
- If GPU memory is limited, use gradient accumulation and keep the effective batch size aligned with the plan in Part D.
- Use a small number of data loader workers (e.g., 2-4) and enable `pin_memory` when using a GPU to improve input throughput.
- Record the Colab GPU type, runtime duration, and peak memory usage in tuning_logs/ for each trial.
- Save checkpoints and tuning logs frequently to avoid data loss from runtime disconnects.

## **Part F) Evaluate the Fine-Tuned Models**

Evaluate the best fine-tuned version of each model on the same held-out test set used in Lab 7. Use task-appropriate metrics (refer to the metrics table in the Discussion). Generate the following deliverables:

- Training curves (loss and primary metric over epochs) for each fine-tuned model. Save to outputs/figures/.
- Updated confusion matrix (classification), updated per-class AP report (detection), or updated per-class IoU report (segmentation) for each model.
- At least six qualitative comparison images per model that show the same test sample before and after fine-tuning, with the prediction overlaid (label / box / mask).
- A clear before-vs-after delta for each metric, expressed as both absolute change and percentage change.

## **Part G) Generate the Comparison Summary**

Consolidate baseline (Lab 7) and fine-tuned (Lab 8) performance into a single comparison table. Save the table as outputs/tables/lab08_finetuning_comparison.csv and reproduce it in the report. Use task-appropriate primary metrics - accuracy for classification models, mAP for detection models, and mIoU for segmentation models.

| **Metric**                                            | **Top Model #1 (Before)** | **Top Model #1 (After)** | **Top Model #2 (Before)** | **Top Model #2 (After)** |
| ----------------------------------------------------- | ------------------------- | ------------------------ | ------------------------- | ------------------------ |
| **Primary task metric (accuracy / mAP / mIoU)**       |                           |                          |                           |                          |
| **Secondary task metric (F1 / AP@\[.5:.95\] / Dice)** |                           |                          |                           |                          |
| **Per-class / per-category breakdown**                |                           |                          |                           |                          |
| **Validation loss (best)**                            |                           |                          |                           |                          |
| **Train-Validation gap (overfitting indicator)**      |                           |                          |                           |                          |
| **Training time (s)**                                 |                           |                          |                           |                          |
| **Inference time per sample (ms)**                    |                           |                          |                           |                          |
| **Trainable parameter count**                         |                           |                          |                           |                          |
| **Improvement vs. baseline (Δ, %)**                   |                           |                          |                           |                          |

In addition, generate a grouped bar chart that visualizes the before-vs-after performance for both models, save it as outputs/figures/lab08_before_after_comparison.png, and embed it in the Results section of the report.

**RESULTS AND DISCUSSION**

## **A. Diagnosis Summary**

For each of the two selected models, summarize the failure modes that the baseline exhibited and the rationale for the chosen fine-tuning strategy. Reference the diagnostic figures from Part C and explain how each strategy in the fine-tuning plan was expected to address a specific weakness.

## **B. Fine-Tuning Trials**

Report the trials performed per model: the configuration tested, the validation metric obtained, and the trial selected as best. Discuss whether the search space was adequately explored and which lever produced the largest single-trial improvement.

## **C. Quantitative Improvement**

Compare baseline and fine-tuned performance using the consolidated table from Part G. Quantify the absolute and percentage improvement on the primary task metric. Interpret whether the improvement is meaningful given the size of the test set and any expected variance from random initialization.

## **D. Qualitative Improvement**

Discuss the qualitative comparison images. Identify the categories or sample types where fine-tuning had the largest visible effect and any categories where no improvement (or regression) was observed.

## **E. Computational Cost of Fine-Tuning**

Compare the additional training time, parameter footprint, and inference latency of each fine-tuned model against its baseline. Discuss whether the gains justify the extra cost in the context of the deployment scenario you envision.

## **F. Identified Limitations**

Discuss at least three limitations observed during fine-tuning. Consider: dataset-size constraints on the strength of fine-tuning, hyperparameters that could not be explored due to compute limits, residual misclassification or detection / segmentation errors that survived all trials, and any signs of overfitting or catastrophic forgetting introduced by the fine-tuning process.

# **Questions (Answer Individually)**

- Distinguish between training a model from scratch, transfer learning, and fine-tuning. Why is fine-tuning typically preferred over training from scratch when the target dataset is small?
- Explain why fine-tuning generally uses a smaller learning rate than initial training, and why differential (layer-wise) learning rates often outperform a single global learning rate.
- If the two selected models target different task types (e.g., one classification model and one detection model), why is it incorrect to compare them using a single shared metric? Justify which metric you used for each model and why it is the right choice.
- Describe one experimental change you applied during fine-tuning that produced a measurable improvement, and one that did not. Using your tuning logs as evidence, propose an explanation for both outcomes.
- Based on your before-vs-after comparison, which of the two fine-tuned models would you recommend for deployment? Justify your recommendation using at least three specific pieces of evidence from your results including at least one accuracy-based metric and at least one cost-based metric (training time, parameter count, or inference latency).

Write a conclusion of 8-10 sentences in paragraph form that summarizes:

**CONCLUSION**

- The criteria used to select the two best models from Lab 7 and how those criteria were applied.
- The dominant failure modes diagnosed in each baseline and the fine-tuning strategies designed to address them.
- The quantitative improvement observed for each model after fine-tuning, supported by task-appropriate metrics.
- The qualitative differences observed between baseline and fine-tuned predictions on representative test samples.
- The trade-offs between the achieved performance gain and the additional computational cost incurred.
- The remaining limitations and the recommended direction for any further refinement work in subsequent laboratories.