# Reflection Questions

## Biggest jump
The biggest jump came when I fixed the learning rate from 0.00001 in Round 1 to 0.001 in Round 2d. Test accuracy moved from 24.66% to 79.81%, which is a gain of 55.15 percentage points. I think this happened because the first learning rate was too small, so the model barely updated in two epochs. At 0.001, the updates were strong enough to learn quickly but still stable.

## Underfitting vs. overfitting
Underfitting means the model is too weak or not trained enough, so both training and validation are low. Overfitting means training gets very high but validation stops improving, so the model memorizes more than it generalizes. In Round 1, we saw underfitting because train accuracy was 18.69% and validation was 24.60%. In Round 5a, we saw overfitting signs because train reached 99.02% while validation was 91.10%, giving a 7.92% gap.

## Why transfer learning works
Transfer learning works because early CNN layers learn general visual features like edges, corners, and simple textures. Those features are still useful even if the new dataset is grayscale clothing instead of ImageNet objects. In our runs, this helped us move from weak baseline performance to around 91% test accuracy after tuning. So even when the final classes are different, pretrained filters still give a strong starting point.

## Fine-tuning learning rate
When the backbone is frozen, only a small head is trained, so larger learning rates are usually okay. After unfreezing, all 2,236,682 parameters become trainable, so aggressive updates can damage pretrained knowledge. That is why a smaller learning rate worked better for fine-tuning in our experiment. We saw this clearly when 4c (LR=0.0001) reached 89.48%, though 4b (LR=0.001) was slightly higher at 89.88% but less stable.

## Confusion matrix
The two classes confused most often were Shirt and T-shirt. The confusion matrix showed 124 samples of Shirt predicted as T-shirt and 64 samples of T-shirt predicted as Shirt. This makes sense because these two classes have very similar shapes and surface patterns in Fashion-MNIST. To improve this, I would add class-focused augmentation and spend more training effort on samples where these two labels overlap.

## Next dataset
If I get a new dataset tomorrow, first I will inspect class balance, image quality, and train/validation/test split quality. Second, I will run a frozen-backbone transfer learning baseline to get a fast reference score and check if the pipeline is working. Third, I will fine-tune with a smaller learning rate, add augmentation, and compare schedulers while watching train-val gaps and confusion patterns. This order is practical because it gives a reliable baseline before spending time on heavier tuning.
