# VAD-wav2vec2

## Features

- Fine-tunes classifier-head for Wav2Vec2 model for binary classification (speech/non-speech)
- Processes audio in fixed-length windows with configurable stride
- Handles class imbalance using weighted loss
- Training visualization with loss and F1 curves

## Files

- `train.py`: Main training script with model setup, training loop, and evaluation
- `VAD_dataset.py`: Custom dataset class for loading and preprocessing audio data

## Usage

1. Prepare your data in JSON manifest format with audio file paths and RTTM annotations
2. Update paths in train.py to point to your:
  - Wav2Vec2 checkpoint
  - Training manifest
  - Validation manifest

3. Run training:
```bash
python train.py
