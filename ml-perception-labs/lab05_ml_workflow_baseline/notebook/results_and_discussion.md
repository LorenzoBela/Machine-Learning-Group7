## Results and Discussion

### A. Baseline Model Performance

The Logistic Regression baseline achieves modest accuracy on this 5-class relief goods classification task. Unlike k-NN (which is a lazy learner that stores data and computes distances at prediction time, resulting in near-instant "training"), Logistic Regression actively solves an optimization problem to learn class-separating weights. This produces meaningful training time and a proper learned decision boundary.

### B. Class-wise Behavior

The confusion matrix reveals which classes are most often confused with each other. Classes with similar visual appearance (e.g., similar packaging, colors, or shapes) tend to have lower precision/recall. This is expected when using flattened pixel features, which lose all spatial structure.

### C. Error Characteristics

Common misclassification patterns include:
- Classes with similar color distributions being confused
- Objects photographed at different angles or scales being misclassified
- Background clutter affecting pixel-based features

### D. Limitations of Baseline Approach

Flattened pixel features fundamentally limit perception because they:
- Destroy spatial relationships between pixels
- Are not translation-invariant (same object shifted slightly = completely different feature vector)
- Cannot capture hierarchical visual patterns (edges → textures → parts → objects)
- Result in very high-dimensional, sparse feature spaces

### E. Motivation for CNNs

CNNs address these weaknesses by:
- Using convolutional filters that preserve spatial structure
- Sharing weights across spatial locations (translation invariance)
- Learning hierarchical feature representations automatically
- Reducing dimensionality through pooling layers

## Questions

**1. Why is it important to establish a baseline model?**

A baseline model provides a minimum performance reference that helps determine whether a problem truly requires complex models (like deep learning) or whether simpler approaches suffice. It reveals dataset difficulty, feature separability, and expected performance bounds. Without a baseline, there is no way to quantify the improvement that more sophisticated models provide.

**2. What information does a confusion matrix provide beyond accuracy?**

A confusion matrix shows per-class performance, revealing which specific classes are being confused with each other. While accuracy gives a single aggregate number, the confusion matrix exposes patterns like systematic misclassifications between similar classes, class imbalance effects, and whether errors are concentrated in specific categories.

**3. Why do classical ML models struggle with image data?**

Classical ML models require fixed-length, handcrafted feature vectors and cannot learn spatial hierarchies. They treat each pixel independently, losing critical spatial relationships. They are also not invariant to common image transformations (translation, rotation, scale), making them sensitive to the exact position and orientation of objects.

**4. How does feature representation affect model performance?**

Feature representation is arguably more important than model choice. Raw flattened pixels produce high-dimensional, noisy features where meaningful patterns are buried. Better representations (like HOG, color histograms, or CNN-learned features) capture more discriminative information, directly improving model performance without changing the classifier.

**5. When would a baseline model be sufficient for a perception task?**

A baseline model may suffice when classes are visually very distinct (different colors, shapes, sizes), the dataset is small enough that deep learning would overfit, computational resources are limited, or interpretability is more important than raw accuracy. If a simple model achieves near-perfect accuracy, the problem may not require deep learning at all.

## Conclusion

This lab established an end-to-end machine learning workflow for image classification using a Logistic Regression baseline on the Relief Goods dataset.

**Key takeaways:**
- The complete ML workflow (data preparation → feature extraction → training → validation → testing → interpretation) provides a systematic framework for developing perception systems.
- Logistic Regression, while simple, actively learns a decision boundary during training (unlike lazy learners like k-NN), producing meaningful training time and interpretable weights.
- Flattened pixel features, even with StandardScaler normalization, fundamentally limit classification performance because they destroy spatial structure and are sensitive to pixel-level variations.
- The modest baseline accuracy motivates the transition to Convolutional Neural Networks (CNNs), which can learn hierarchical, spatially-aware features directly from images.
- Establishing this baseline is essential before applying deep learning, as it quantifies the improvement that learned representations provide over handcrafted features.