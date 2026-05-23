# Lab 08 Fine-Tuning: Results, Discussion, and Conclusion

## 1. Results Summary

After increasing the dataset to 100 images per class, the notebook used a 60/20/20 split, giving **300 training samples**, **100 validation samples**, and **100 test samples**. The completed fine-tuning run produced the following trial results.

| Model | Trial ID | Best Epoch | Val Accuracy | Test Accuracy | F1 Macro | Training Time (s) | Inference Time (ms/sample) | Peak GPU Memory (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | m1_t1 | 2 | 82.00% | 65.00% | 65.29% | 108.33 | 0.229 | **103.80** |
| **MobileNetV2** | m1_t2 | 2 | 82.00% | 66.00% | 66.74% | **105.12** | 0.272 | 289.10 |
| **MobileNetV2** | m1_t3 | 3 | **84.00%** | **72.00%** | **72.69%** | 165.13 | 0.229 | 281.69 |
| **EfficientDetLite0** | m2_t1 | 1 | 56.00% | 53.00% | 53.00% | **77.53** | 0.177 | **119.27** |
| **EfficientDetLite0** | m2_t2 | 5 | **58.00%** | **57.00%** | **56.47%** | 159.29 | **0.171** | 314.19 |
| **EfficientDetLite0** | m2_t3 | 1 | **58.00%** | 50.00% | 48.12% | 99.27 | 0.177 | 303.36 |

Best before/after comparison from `outputs/tables/lab08_finetuning_comparison.csv`:

| Model | Baseline Accuracy | Fine-Tuned Accuracy | Absolute Gain | Relative Gain | Baseline F1 Macro | Fine-Tuned F1 Macro |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MobileNetV2 | 69.00% | **72.00%** | +3.00 pts | +4.35% | 68.92% | **72.69%** |
| EfficientDetLite0 | 54.00% | **57.00%** | +3.00 pts | +5.56% | 52.71% | **56.47%** |

---

## 2. Discussion

### 2.1 Diagnosis Summary

The larger dataset changed the evaluation surface from the earlier small split to a more stable 100-sample test set. The main failure mode remained the same: visually similar grocery categories, especially cylindrical or packaged items, were still harder to separate than classes with clearer texture or shape cues. Because the augmented samples increased variation in orientation, lighting, and contrast, the models were tested against a slightly broader distribution than before.

### 2.2 Fine-Tuning Trials

**MobileNetV2** remained the stronger architecture. Its best configuration was **m1_t3**, which unfroze the backbone, used a lower learning rate (`1e-5`), cosine scheduling, lighter weight decay, label smoothing, and no extra online augmentation. This trial reached **84.00% validation accuracy** and **72.00% test accuracy**, outperforming the other MobileNetV2 trials by 6-7 percentage points on the test set.

**EfficientDetLite0** improved more modestly. Its best configuration was **m2_t2**, which used full fine-tuning with a `3e-5` learning rate, step scheduler, label smoothing, augmentation, and gradient accumulation. It reached **58.00% validation accuracy** and **57.00% test accuracy**. The result is better than its baseline, but it still trails MobileNetV2 by 15 percentage points on the final test accuracy.

### 2.3 Quantitative Improvement

Both models improved by **+3.00 absolute accuracy points** over their baseline evaluations on the enlarged split. MobileNetV2 increased from **69.00% to 72.00%**, while EfficientDetLite0 increased from **54.00% to 57.00%**. Macro F1 also improved for both models: MobileNetV2 rose from **68.92% to 72.69%**, and EfficientDetLite0 rose from **52.71% to 56.47%**. This suggests the fine-tuning helped class balance rather than only improving one dominant class.

### 2.4 Qualitative Improvement

The saved confusion matrices and qualitative comparison figures show that the fine-tuned models became more consistent, but not perfect. MobileNetV2 produced the clearest improvement and was better at preserving class-level separation. EfficientDetLite0 still showed more uncertainty, which is expected because the EfficientDet family is originally designed around detection-style features rather than being a simple classification-first architecture.

### 2.5 Computational Cost

MobileNetV2 `m1_t3` was the most accurate trial, but it also took the longest among MobileNetV2 runs at **165.13 seconds**. Its inference time stayed very low at **0.229 ms/sample**, matching `m1_t1` and remaining practical for real-time use. EfficientDetLite0 `m2_t2` took **159.29 seconds** and used the most GPU memory among all trials at **314.19 MB**, but its inference time was the fastest at **0.171 ms/sample**. Overall, the resource cost stayed lightweight for both models, but MobileNetV2 delivered a better accuracy-to-cost tradeoff.

### 2.6 Limitations

The dataset is still small even after augmentation: 500 total images is useful for a lab-scale experiment, but it is not enough to guarantee robust real-world performance. The added images are augmented versions of existing samples, so they increase variation but do not fully replace new real captures. The validation accuracy for MobileNetV2 was higher than its test accuracy, which indicates that some distribution gap or residual overfitting may still exist.

---

## 3. Conclusion

Fine-tuning improved both selected Lab 7 models after the dataset was expanded to 100 images per class. MobileNetV2 achieved the best final result with **72.00% test accuracy** and **72.69% macro F1** using trial **m1_t3**. EfficientDetLite0 improved to **57.00% test accuracy** and **56.47% macro F1** using trial **m2_t2**, but it remained clearly behind MobileNetV2. The best MobileNetV2 model also kept inference extremely fast at **0.229 ms/sample**, making it the better candidate for deployment. EfficientDetLite0 had slightly faster inference in its best trial, but the accuracy tradeoff was too large for this classification task. The experiment shows that careful fine-tuning and dataset expansion can improve generalization, but the gains are still limited by the size and diversity of the dataset. Future work should add more real images per class instead of relying mainly on augmented copies. The final recommendation is to use **MobileNetV2 trial m1_t3** as the best-performing fine-tuned model.
