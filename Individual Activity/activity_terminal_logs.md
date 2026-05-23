# Terminal Logs by Run

## Run 1

```text
========================================================================================
Running 1
Config: {'run_id': '1', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 1e-05, 'batch_size': 64, 'epochs': 2, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     2.3265    14.40%    2.2333   18.40%     5.0s
     2     2.2174    18.69%    2.1225   24.60%     4.8s
--------------------------------------------------------
Training time : 9.8 seconds
TEST accuracy : 24.66%
UNDERFITTING warning: final train acc is below 50%
```

## Run 7

```text
========================================================================================
Running 7
Config: {'run_id': '7', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 10, 'optimizer': 'adam', 'scheduler': 'step', 'train_size': 10000, 'use_augmentation': True}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.6478    78.43%    0.3558   88.10%    23.0s
     2     0.3269    88.22%    0.2794   90.40%    25.6s
     3     0.2639    90.41%    0.2315   91.70%    18.4s
     4     0.2119    92.23%    0.2253   92.00%    15.8s
     5     0.1848    93.43%    0.2034   93.20%    15.7s
     6     0.1699    93.72%    0.2025   93.60%    28.0s
     7     0.1453    94.63%    0.1942   93.60%    20.0s
     8     0.1309    95.39%    0.1939   93.70%    20.2s
     9     0.1187    95.58%    0.1989   93.40%    18.8s
    10     0.1076    95.94%    0.1922   94.00%    26.8s
--------------------------------------------------------
Training time : 212.4 seconds
TEST accuracy : 91.71%
```

## Run 2a

```text
========================================================================================
Running 2a
Config: {'run_id': '2a', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 1.0, 'batch_size': 64, 'epochs': 3, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1   144.4548    63.42%   66.4811   74.40%     4.3s
     2    68.2554    74.91%   64.4438   74.80%     4.3s
     3    62.5694    76.42%   81.4613   75.00%     4.3s
--------------------------------------------------------
Training time : 12.9 seconds
TEST accuracy : 75.33%
```

## Run 2b

```text
========================================================================================
Running 2b
Config: {'run_id': '2b', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.1, 'batch_size': 64, 'epochs': 3, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1    14.1335    63.47%    7.3294   74.20%     4.6s
     2     6.5517    75.09%    6.3254   78.20%     4.3s
     3     5.7065    77.80%    9.9481   70.40%     4.3s
--------------------------------------------------------
Training time : 13.2 seconds
TEST accuracy : 70.97%
```

## Run 2c

```text
========================================================================================
Running 2c
Config: {'run_id': '2c', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.01, 'batch_size': 64, 'epochs': 3, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     1.3000    68.36%    0.9286   75.40%     4.2s
     2     0.8667    76.22%    0.8305   78.80%     6.9s
     3     0.7302    79.80%    0.9710   76.40%     7.4s
--------------------------------------------------------
Training time : 18.5 seconds
TEST accuracy : 77.73%
```

## Run 2d

```text
========================================================================================
Running 2d
Config: {'run_id': '2d', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 3, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     1.0518    66.82%    0.6621   80.00%     4.3s
     2     0.6133    79.09%    0.6014   80.20%     6.5s
     3     0.5356    81.49%    0.6036   79.00%     4.3s
--------------------------------------------------------
Training time : 15.1 seconds
TEST accuracy : 79.81%
```

## Run 2e

```text
========================================================================================
Running 2e
Config: {'run_id': '2e', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 3, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     1.9570    35.80%    1.5629   59.40%     4.5s
     2     1.4003    62.20%    1.1818   70.00%     4.4s
     3     1.1232    69.62%    1.0112   72.20%     4.4s
--------------------------------------------------------
Training time : 13.3 seconds
TEST accuracy : 73.72%
```

## Run 3a

```text
========================================================================================
Running 3a
Config: {'run_id': '3a', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 3, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     1.0518    66.82%    0.6621   80.00%     5.2s
     2     0.6133    79.09%    0.6014   80.20%     5.1s
     3     0.5356    81.49%    0.6036   79.00%     4.9s
--------------------------------------------------------
Training time : 15.2 seconds
TEST accuracy : 79.81%
```

## Run 3b

```text
========================================================================================
Running 3b
Config: {'run_id': '3b', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 8, 'optimizer': 'adam', 'scheduler': None, 'train_size': 5000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     1.0518    66.82%    0.6621   80.00%     7.1s
     2     0.6133    79.09%    0.6014   80.20%     4.6s
     3     0.5356    81.49%    0.6036   79.00%     5.3s
     4     0.4939    82.89%    0.5545   81.20%     4.4s
     5     0.4668    83.56%    0.5654   81.80%     4.2s
     6     0.4313    84.67%    0.5879   80.00%     4.0s
     7     0.4416    84.96%    0.5493   81.60%     4.7s
     8     0.4108    85.04%    0.5646   80.40%     4.3s
--------------------------------------------------------
Training time : 38.5 seconds
TEST accuracy : 80.87%
```

## Run 3c

```text
========================================================================================
Running 3c
Config: {'run_id': '3c', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 5, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.8312    72.66%    0.5314   81.20%    12.3s
     2     0.5551    81.10%    0.4702   84.60%    11.8s
     3     0.5009    82.29%    0.4588   84.40%    11.0s
     4     0.4741    83.30%    0.4524   83.70%     9.0s
     5     0.4662    83.22%    0.4628   84.30%     9.1s
--------------------------------------------------------
Training time : 53.1 seconds
TEST accuracy : 81.85%
```

## Run 3d

```text
========================================================================================
Running 3d
Config: {'run_id': '3d', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 5, 'optimizer': 'adam', 'scheduler': None, 'train_size': 20000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.7013    76.47%    0.5360   81.50%    17.8s
     2     0.5199    81.53%    0.5032   82.00%    20.7s
     3     0.4889    82.62%    0.4915   82.15%    17.4s
     4     0.4774    83.04%    0.4769   83.70%    18.6s
     5     0.4716    82.99%    0.4848   82.60%    17.9s
--------------------------------------------------------
Training time : 92.4 seconds
TEST accuracy : 82.79%
```

## Run 4a

```text
========================================================================================
Running 4a
Config: {'run_id': '4a', 'freeze_backbone': True, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 5, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.8312    72.66%    0.5314   81.20%    11.2s
     2     0.5551    81.10%    0.4702   84.60%     8.8s
     3     0.5009    82.29%    0.4588   84.40%     9.0s
     4     0.4741    83.30%    0.4524   83.70%     8.8s
     5     0.4662    83.22%    0.4628   84.30%     8.8s
--------------------------------------------------------
Training time : 46.6 seconds
TEST accuracy : 81.85%
```

## Run 4b

```text
========================================================================================
Running 4b
Config: {'run_id': '4b', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 5, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.4910    82.76%    0.3134   89.40%    14.3s
     2     0.3004    89.68%    0.3948   86.90%    19.2s
     3     0.2606    90.82%    0.2599   91.80%    19.7s
     4     0.2170    92.33%    0.3009   90.00%    14.5s
     5     0.1837    93.50%    0.2579   91.30%    14.4s
--------------------------------------------------------
Training time : 82.1 seconds
TEST accuracy : 89.88%
```

## Run 4c

```text
========================================================================================
Running 4c
Config: {'run_id': '4c', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 5, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.5662    81.11%    0.3112   89.20%    20.0s
     2     0.2551    90.93%    0.2375   91.70%    14.7s
     3     0.1519    94.72%    0.2381   91.30%    17.0s
     4     0.0999    96.44%    0.2633   91.20%    15.1s
     5     0.0724    97.57%    0.2915   90.70%    19.7s
--------------------------------------------------------
Training time : 86.4 seconds
TEST accuracy : 89.48%
```

## Run 4d

```text
========================================================================================
Running 4d
Config: {'run_id': '4d', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 1e-05, 'batch_size': 64, 'epochs': 5, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     1.5227    53.96%    0.8799   80.00%    16.9s
     2     0.7288    80.16%    0.5385   83.90%    13.6s
     3     0.5157    84.39%    0.4266   86.20%    14.0s
     4     0.4153    86.70%    0.3641   87.80%    19.5s
     5     0.3580    87.90%    0.3338   88.90%    17.2s
--------------------------------------------------------
Training time : 81.1 seconds
TEST accuracy : 87.04%
```

## Run 5a

```text
========================================================================================
Running 5a
Config: {'run_id': '5a', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 8, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': False}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.5662    81.11%    0.3112   89.20%    14.4s
     2     0.2551    90.93%    0.2375   91.70%    16.6s
     3     0.1519    94.72%    0.2381   91.30%    16.3s
     4     0.0999    96.44%    0.2633   91.20%    13.4s
     5     0.0724    97.57%    0.2915   90.70%    13.6s
     6     0.0432    98.70%    0.2886   90.90%    16.2s
     7     0.0312    99.03%    0.3155   91.00%    13.5s
     8     0.0302    99.02%    0.3148   91.10%    13.7s
--------------------------------------------------------
Training time : 117.7 seconds
TEST accuracy : 90.24%
```

## Run 5b

```text
========================================================================================
Running 5b
Config: {'run_id': '5b', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 8, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': True}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.6478    78.43%    0.3558   88.10%    17.8s
     2     0.3269    88.22%    0.2794   90.40%    15.7s
     3     0.2639    90.41%    0.2315   91.70%    15.2s
     4     0.2295    91.59%    0.2222   92.00%    17.7s
     5     0.1949    92.78%    0.2086   93.00%    17.7s
     6     0.1779    93.61%    0.2364   92.30%    15.8s
     7     0.1512    94.51%    0.2077   93.30%    15.2s
     8     0.1399    94.99%    0.2149   92.90%    15.9s
--------------------------------------------------------
Training time : 130.9 seconds
TEST accuracy : 91.48%
```

## Run 5c

```text
========================================================================================
Running 5c
Config: {'run_id': '5c', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 12, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': True}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.6478    78.43%    0.3558   88.10%    15.1s
     2     0.3269    88.22%    0.2794   90.40%    20.4s
     3     0.2639    90.41%    0.2315   91.70%    19.2s
     4     0.2295    91.59%    0.2222   92.00%    18.8s
     5     0.1949    92.78%    0.2086   93.00%    27.3s
     6     0.1779    93.61%    0.2364   92.30%    16.8s
     7     0.1512    94.51%    0.2077   93.30%    19.0s
     8     0.1399    94.99%    0.2149   92.90%    22.0s
     9     0.1176    95.86%    0.2120   93.00%    21.9s
    10     0.1043    95.94%    0.2223   92.70%    29.2s
    11     0.1009    96.37%    0.2164   92.70%    16.9s
    12     0.0857    96.83%    0.2379   92.80%    16.2s
--------------------------------------------------------
Training time : 242.7 seconds
TEST accuracy : 91.41%
```

## Run 6a

```text
========================================================================================
Running 6a
Config: {'run_id': '6a', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 10, 'optimizer': 'adam', 'scheduler': None, 'train_size': 10000, 'use_augmentation': True}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.6478    78.43%    0.3558   88.10%    21.6s
     2     0.3269    88.22%    0.2794   90.40%    19.2s
     3     0.2639    90.41%    0.2315   91.70%    19.1s
     4     0.2295    91.59%    0.2222   92.00%    16.8s
     5     0.1949    92.78%    0.2086   93.00%    17.5s
     6     0.1779    93.61%    0.2364   92.30%    16.5s
     7     0.1512    94.51%    0.2077   93.30%    16.9s
     8     0.1399    94.99%    0.2149   92.90%    16.0s
     9     0.1176    95.86%    0.2120   93.00%    27.8s
    10     0.1043    95.94%    0.2223   92.70%    16.5s
--------------------------------------------------------
Training time : 188.1 seconds
TEST accuracy : 91.46%
```

## Run 6b

```text
========================================================================================
Running 6b
Config: {'run_id': '6b', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 10, 'optimizer': 'adam', 'scheduler': 'step', 'train_size': 10000, 'use_augmentation': True}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.6478    78.43%    0.3558   88.10%    22.3s
     2     0.3269    88.22%    0.2794   90.40%    22.5s
     3     0.2639    90.41%    0.2315   91.70%    16.2s
     4     0.2119    92.23%    0.2253   92.00%    23.9s
     5     0.1848    93.43%    0.2034   93.20%    28.3s
     6     0.1699    93.72%    0.2025   93.60%    35.2s
     7     0.1453    94.63%    0.1942   93.60%    34.8s
     8     0.1309    95.39%    0.1939   93.70%    34.4s
     9     0.1187    95.58%    0.1989   93.40%    33.6s
    10     0.1076    95.94%    0.1922   94.00%    49.1s
--------------------------------------------------------
Training time : 300.3 seconds
TEST accuracy : 91.71%
```

## Run 6c

```text
========================================================================================
Running 6c
Config: {'run_id': '6c', 'freeze_backbone': False, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 64, 'epochs': 10, 'optimizer': 'adam', 'scheduler': 'cosine', 'train_size': 10000, 'use_augmentation': True}
 Epoch  TrainLoss  TrainAcc   ValLoss   ValAcc     Time
--------------------------------------------------------
     1     0.6478    78.43%    0.3558   88.10%    30.2s
     2     0.3264    88.38%    0.2796   90.30%    29.2s
     3     0.2624    90.47%    0.2337   91.90%    33.8s
     4     0.2256    91.60%    0.2287   91.70%    34.9s
     5     0.1867    93.23%    0.2064   93.00%    32.5s
     6     0.1620    93.92%    0.1952   93.00%    31.9s
     7     0.1397    94.80%    0.1911   93.70%    33.0s
     8     0.1220    95.66%    0.1851   94.10%    38.1s
     9     0.1056    96.29%    0.1890   94.10%    35.4s
    10     0.1002    96.28%    0.1855   94.20%   117.3s
--------------------------------------------------------
Training time : 416.3 seconds
TEST accuracy : 91.92%
```
