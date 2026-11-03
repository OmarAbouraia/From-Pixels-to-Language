# CNN V3 — Final Results

## Model
- Architecture: Residual CNN trained from scratch
- Dataset: CUB-200 (200 bird species)
- Parameters: 11,308,424
- Training epochs: 90
- Selected checkpoint: Epoch 89
- Optimizer: AdamW
- Learning rate: 0.003
- Scheduler: 5-epoch warmup + cosine decay

## Final Performance
- Best validation accuracy: 48.83%
- Test accuracy: 50.64%
- Correct test predictions: 2,934 / 5,794
- Macro precision: 51.87%
- Macro recall: 50.76%
- Macro F1-score: 50.52%
- Weighted precision: 52.16%
- Weighted recall: 50.64%
- Weighted F1-score: 50.62%

## Analysis
Training accuracy approached 100%, while validation
accuracy reached 48.83%, indicating substantial overfitting.

Despite this gap, the model achieved 50.64% accuracy
on the held-out test dataset.

The best model was selected using validation accuracy,
then evaluated on the test dataset without further training.
