# Results and Discussion

## A. Model Architecture Analysis
We used four models in this lab: EfficientDetLite0, MobileNetV2, YOLOv8Nano, and YOLOv10Nano. We trained all of them with the same preprocessing steps and the same data split so the comparison is fair. Their parameter counts were 3,377,413 (EfficientDetLite0), 2,230,277 (MobileNetV2), 3,011,807 (YOLOv8Nano), and 2,708,974 (YOLOv10Nano). MobileNetV2 had the fewest parameters but still gave the best overall test performance in our run.

## B. Training Behavior
The validation results show clear differences in how the models learned. MobileNetV2 got the highest Best Val Accuracy at 0.8108 and was marked N for overfitting. EfficientDetLite0 got 0.6216 (N), YOLOv8Nano got 0.5946 (Y), and YOLOv10Nano got 0.3784 (N). The overfitting mark on YOLOv8Nano matches its weaker test performance compared to its validation peak.

## C. Performance Evaluation
Using the notebook comparison table, the test metrics were:

- EfficientDetLite0: Accuracy 0.5263, Precision(macro) 0.4901, Recall(macro) 0.4909, F1(macro) 0.4768
- MobileNetV2: Accuracy 0.5789, Precision(macro) 0.5828, Recall(macro) 0.5709, F1(macro) 0.5429
- YOLOv8Nano: Accuracy 0.3684, Precision(macro) 0.4633, Recall(macro) 0.3764, F1(macro) 0.3574
- YOLOv10Nano: Accuracy 0.2895, Precision(macro) 0.1273, Recall(macro) 0.2400, F1(macro) 0.1430

From these values, MobileNetV2 ranked first in all major test metrics (accuracy, macro precision, macro recall, and macro F1). EfficientDetLite0 ranked second, YOLOv8Nano ranked third, and YOLOv10Nano ranked fourth in this run.

## D. Computational Cost Analysis
Training times were 554.6 s (EfficientDetLite0), 513.9 s (MobileNetV2), 417.0 s (YOLOv8Nano), and 440.4 s (YOLOv10Nano). YOLOv8Nano was the fastest to train, but MobileNetV2 gave better prediction results. Looking at both speed and performance, MobileNetV2 gave the best balance for this experiment.

## E. Identified Limitations
1. The dataset is not very large, so results can still change between runs.
2. Some classes look very similar, so misclassifications still happen.
3. We used mostly the same training settings across models, which may not be best for each one.
4. The YOLO-based models were adapted for classification, which may limit their performance.
5. Lighting, angle, and background differences in the images can hurt generalization.

# Questions (Answer Individually)

**Why is it important to use the same dataset split (same random seed, same proportions) when comparing multiple models? What types of errors can arise if splits differ between models?**  
Using the same split makes sure every model is tested on the same images. That keeps the comparison fair because only the model is changing, not the data. In our results, the gap is big (best accuracy 0.5789 vs lowest 0.2895), so different splits could change the ranking a lot. If splits are different, we might pick the wrong model just because it got an easier test set.

**Explain the No Free Lunch Theorem in your own words. How does this theorem motivate the multi-model comparison approach used in this laboratory?**  
In simple terms, No Free Lunch means there is no single model that is best for all datasets. That is why we tested multiple models instead of trusting only one from the start. In our run, YOLOv8Nano trained faster (417.0 s) than MobileNetV2 (513.9 s), but MobileNetV2 still got much better test accuracy (0.5789 vs 0.3684). This shows why side-by-side testing is important before deployment.

**For imbalanced datasets, why is F1-score (macro-averaged) a more informative metric than raw accuracy? Provide an example using class distributions from your own dataset.**  
Macro F1 is useful for imbalanced data because it gives equal importance to each class. Accuracy alone can look okay even when the model is doing poorly on smaller classes. In our results, YOLOv10Nano has 0.2895 accuracy but only 0.1430 macro F1, which shows weak class-level performance. Even for MobileNetV2, accuracy is 0.5789 while macro F1 is 0.5429, so macro F1 gives extra insight.

**Describe the key difference between a classical ML model (e.g., SVM or Random Forest) and a deep neural network in terms of feature learning. What is the role of feature engineering in each approach?**  
Classical ML usually depends on hand-made or precomputed features, while deep neural networks learn features directly from images. In our notebook flow, we made flattened feature arrays for classical-style inputs, which shows how important feature setup is in that approach. Deep models learn their own internal features during training, so architecture and optimization have a big impact on performance. So feature engineering matters more in classical ML, while deep learning focuses more on learned representations.

**Based on your experimental results, which of the four models would you recommend for deployment? Justify your choice using at least three specific pieces of evidence from your comparison table and confusion matrices.**  
I would recommend MobileNetV2 for deployment in this run. It has the best test accuracy (0.5789), best macro precision (0.5828), best macro recall (0.5709), and best macro F1 (0.5429). It also has the highest Best Val Accuracy (0.8108) with overfitting marked N, which suggests better stability. On top of that, it has the smallest parameter count (2,230,277), so it is a strong choice for both performance and efficiency.

# Conclusion

Based on our experiment, MobileNetV2 is the most suitable model for this dataset. It achieved the highest test accuracy (0.5789), precision (0.5828), recall (0.5709), and macro F1-score (0.5429) among the four models. It also reached the best validation accuracy (0.8108) without being flagged for overfitting. In addition, it had the smallest parameter count (2,230,277), which makes it efficient for deployment. EfficientDetLite0 performed second best and remained a strong alternative, but its test metrics were consistently lower than MobileNetV2. YOLOv8Nano trained the fastest, yet its lower final accuracy and overfitting flag reduce its reliability on unseen data. YOLOv10Nano produced the weakest scores overall, indicating that this architecture needs further tuning for this task. Overall, using the same preprocessing pipeline and fixed dataset split gave us a fair comparison and a clear evidence-based model recommendation.