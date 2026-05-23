# Results and Discussion

## A. CNN Architecture Analysis
The three CNN models are different in depth and structure, and this can be seen in their accuracy and parameter counts.

| Model | Parameters | Best Val Acc | Test Acc |
|---|---:|---:|---:|
| ShallowCNN | 4,215,237 | 0.8919 | 0.9737 |
| RegularizedCNN | 4,291,077 | 0.9459 | 0.9737 |
| DeepCNN | 1,201,445 | 0.9730 | 0.9474 |

ShallowCNN has 2 convolution blocks and a large dense head after flattening. It can learn quickly, but it also tends to memorize the training set because it has no strong regularization.

RegularizedCNN adds one more block and uses BatchNorm and Dropout. This gave better validation performance than ShallowCNN, with best validation accuracy improving from 0.8919 to 0.9459.

DeepCNN uses skip connections and Global Average Pooling. Even with fewer parameters than the other two models, it reached the highest best validation accuracy (0.9730), showing that a better design can beat a bigger model.

## B. Training Behavior
The training curves show clear differences in how each model learned:

- ShallowCNN: it improved fast early on (epoch 10: train 0.9086, val 0.8649), but later train accuracy kept increasing while validation stayed around 0.84 to 0.89. This suggests overfitting starting around epochs 10 to 12.
- RegularizedCNN: it had the most stable pattern, with train and validation staying close through most of training. By epoch 30, it reached train 0.9657 and val 0.9459.
- DeepCNN: it started slower but became very strong by mid-training (epoch 15 val 0.9730). It dipped at around epoch 25 (val 0.9189) but recovered at epoch 30 (val 0.9730).

Overall, RegularizedCNN and DeepCNN generalized better than ShallowCNN, while ShallowCNN showed the largest gap between train and validation performance.

## C. Performance Evaluation
All three confusion matrices are strong, and most classes are classified correctly almost all the time.

- ShallowCNN (Test Acc 0.9737): the main mistake is bottled water being predicted as combo (0.10).
- RegularizedCNN (Test Acc 0.9737): very similar result, with the same bottled water to combo confusion (0.10).
- DeepCNN (Test Acc 0.9474): it shows two small confusion pairs, Noodles to canned goods (0.09) and bottled water to Rice (0.10).

This means the hardest cases are classes that look visually similar in packaging, color, or background. Rice and canned goods were mostly classified consistently across models.

## E. Identified Limitations
1. The test set is small (about 38 images), so even one wrong prediction changes test accuracy by around 2.63%.
2. Since the dataset was group-collected, background and camera setup may influence predictions.
3. Training all three models for 30 epochs on CPU takes time, so it is hard to test many settings.
4. Results still depend a lot on choices like learning rate, dropout, and augmentation intensity.
5. Some confusion between similar classes remains, especially bottled water vs combo and Noodles vs canned goods.

# Questions (Answer Individually)

**1. Why are CNNs fundamentally better suited for image data than MLPs?**  
CNNs are better for images because they keep nearby pixels together and learn local visual patterns first. In our run, the common errors were between visually similar classes, like bottled water and combo, which shows spatial cues matter. CNN filters are reused across the image, so they learn these patterns efficiently instead of treating every pixel relation as separate. This helped our CNN models reach high performance, with test accuracy up to 0.9737.

**2. How does the receptive field increase with depth, and how does that support hierarchical learning?**  
As a CNN gets deeper, each later layer can see a larger area of the image. Early layers learn simple details like edges and color changes, while deeper layers combine them into object-level features. In our results, DeepCNN reached the highest best validation accuracy at 0.9730, while ShallowCNN reached 0.8919. This supports that deeper hierarchical features helped classification.

**3. What does Batch Normalization solve, and how did it affect RegularizedCNN versus ShallowCNN?**  
BatchNorm keeps layer outputs in a stable range, which makes training smoother. In our run, RegularizedCNN improved best validation accuracy from 0.8919 (ShallowCNN) to 0.9459. By epoch 30, RegularizedCNN had train 0.9657 and val 0.9459, which is a small gap. ShallowCNN showed a bigger gap earlier (epoch 10: train 0.9086 vs val 0.8649), so BatchNorm helped reduce overfitting signs.

**4. What is a residual connection in DeepCNN, and what gradient problem does it address?**  
A residual connection is a shortcut that adds the block input to its output. This helps gradients pass through deep layers more easily and reduces vanishing gradient problems. In our notebook, DeepCNN used residual blocks and still reached the highest best validation accuracy at 0.9730. It did this with only 1,201,445 parameters, showing the deep model could train effectively without becoming too large.

**5. How is Global Average Pooling different from Flatten + Dense, and why is it useful here?**  
Global Average Pooling turns each feature map into one value, while Flatten + Dense keeps many values and usually builds a much bigger classifier head. In our run, DeepCNN with GAP had 1,201,445 parameters, much lower than ShallowCNN (4,215,237) and RegularizedCNN (4,291,077). Even with fewer parameters, DeepCNN reached the highest best validation accuracy at 0.9730. This is useful for our small dataset because a lighter model is less likely to memorize noise.

# Conclusion

In this lab, the CNN models clearly worked well for image classification. ShallowCNN gave a strong baseline but showed more overfitting compared to the other models. RegularizedCNN had the most stable training behavior and reached the highest test accuracy (0.9737, tied with ShallowCNN), so RegularizedCNN is the model we selected for our dataset. DeepCNN achieved the best validation score with fewer parameters, but its final test accuracy was lower at 0.9474. Even so, a few confusing class pairs remained, likely because some products look very similar in photos. For Lab 8, we plan to improve results by tuning hyperparameters, adjusting augmentation, and testing stronger feature-learning approaches.