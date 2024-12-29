# VAD-wav2vec2

## Features

- fine-tunes classifier-head for Wav2Vec2 model for binary classification (speech/non-speech)
- processes audio in fixed-length windows with configurable stride
- handles class imbalance using weighted loss
- training visualization with loss and F1 curves

## Files

- `train.py`: main training script with model setup, training loop, and evaluation
- `VAD_dataset.py`: custom dataset class for loading and preprocessing audio data


## Run training:
```bash
python train.py
