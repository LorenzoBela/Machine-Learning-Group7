# Activity Results Summary

Generated UTC: 2026-04-21T10:21:04.440109+00:00

## Environment

- Device selected: cuda:0
- Torch: 2.11.0+cu128
- Torchvision: 0.26.0+cu128
- Dataset source used: torchvision
- Dataset note: Using torchvision.datasets.FashionMNIST (download=True).

## ROUND1

| Run | Test | Train | Val | Gap | Time | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 24.66% | 18.69% | 24.60% | -5.91 pp | 9.8s | underfitting; final test accuracy=24.66% (train-val gap -5.91 pp). |

## ROUND2

| Run | Test | Train | Val | Gap | Time | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2a | 75.33% | 76.42% | 75.00% | 1.42 pp | 12.9s | balanced fit; final test accuracy=75.33% (train-val gap 1.42 pp). |
| 2b | 70.97% | 77.80% | 70.40% | 7.40 pp | 13.2s | balanced fit; final test accuracy=70.97% (train-val gap 7.40 pp). |
| 2c | 77.73% | 79.80% | 76.40% | 3.40 pp | 18.5s | balanced fit; final test accuracy=77.73% (train-val gap 3.40 pp). |
| 2d | 79.81% | 81.49% | 79.00% | 2.49 pp | 15.1s | balanced fit; final test accuracy=79.81% (train-val gap 2.49 pp). |
| 2e | 73.72% | 69.62% | 72.20% | -2.58 pp | 13.3s | balanced fit; final test accuracy=73.72% (train-val gap -2.58 pp). |

## ROUND3

| Run | Test | Train | Val | Gap | Time | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3a | 79.81% | 81.49% | 79.00% | 2.49 pp | 15.2s | balanced fit; final test accuracy=79.81% (train-val gap 2.49 pp). |
| 3b | 80.87% | 85.04% | 80.40% | 4.64 pp | 38.5s | balanced fit; final test accuracy=80.87% (train-val gap 4.64 pp). |
| 3c | 81.85% | 83.22% | 84.30% | -1.08 pp | 53.1s | balanced fit; final test accuracy=81.85% (train-val gap -1.08 pp). |
| 3d | 82.79% | 82.99% | 82.60% | 0.39 pp | 92.4s | balanced fit; final test accuracy=82.79% (train-val gap 0.39 pp). |

## ROUND4

| Run | Test | Train | Val | Gap | Time | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4a | 81.85% | 83.22% | 84.30% | -1.08 pp | 46.6s | balanced fit; final test accuracy=81.85% (train-val gap -1.08 pp). |
| 4b | 89.88% | 93.50% | 91.30% | 2.20 pp | 82.1s | balanced fit; final test accuracy=89.88% (train-val gap 2.20 pp). |
| 4c | 89.48% | 97.57% | 90.70% | 6.87 pp | 86.4s | balanced fit; final test accuracy=89.48% (train-val gap 6.87 pp). |
| 4d | 87.04% | 87.90% | 88.90% | -1.00 pp | 81.1s | balanced fit; final test accuracy=87.04% (train-val gap -1.00 pp). |

## ROUND5

| Run | Test | Train | Val | Gap | Time | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 5a | 90.24% | 99.02% | 91.10% | 7.92 pp | 117.7s | balanced fit; final test accuracy=90.24% (train-val gap 7.92 pp). |
| 5b | 91.48% | 94.99% | 92.90% | 2.09 pp | 130.9s | balanced fit; final test accuracy=91.48% (train-val gap 2.09 pp). |
| 5c | 91.41% | 96.83% | 92.80% | 4.03 pp | 242.7s | balanced fit; final test accuracy=91.41% (train-val gap 4.03 pp). |

## ROUND6

| Run | Test | Train | Val | Gap | Time | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 6a | 91.46% | 95.94% | 92.70% | 3.24 pp | 188.1s | balanced fit; final test accuracy=91.46% (train-val gap 3.24 pp). |
| 6b | 91.71% | 95.94% | 94.00% | 1.94 pp | 300.3s | balanced fit; final test accuracy=91.71% (train-val gap 1.94 pp). |
| 6c | 91.92% | 96.28% | 94.20% | 2.08 pp | 416.3s | balanced fit; final test accuracy=91.92% (train-val gap 2.08 pp). |

## ROUND7

| Run | Test | Train | Val | Gap | Time | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 91.71% | 95.94% | 94.00% | 1.94 pp | 212.4s | balanced fit; final test accuracy=91.71% (train-val gap 1.94 pp). |

## Final

- Round 1 test accuracy: 24.66%
- Round 7 test accuracy: 91.71%
- Improvement over Round 1: 67.05 percentage points

### Top Round 7 Confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 124 | Shirt | T-shirt |
| 64 | T-shirt | Shirt |
| 60 | Shirt | Coat |
| 50 | Shirt | Pullover |
| 47 | Coat | Shirt |
| 44 | Pullover | Shirt |
| 43 | Pullover | Coat |
| 37 | Shirt | Dress |
| 33 | Coat | Pullover |
| 33 | Ankle boot | Sneaker |