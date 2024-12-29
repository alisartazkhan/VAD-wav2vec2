# VAD-wav2vec2

* audio files - https://drive.google.com/drive/folders/18oiXthBZBPqf1Sq_rW6VMkwlsRX30YYt?usp=sharing
* rttms - https://drive.google.com/drive/folders/1T7aBhghoROyEZsTOn6Sge8YyGNUzJ3BF?usp=sharing
* Ahmed's wav2vec fairseq model converted to a pytorch model - https://drive.google.com/drive/folders/1wVK5nAAXKgNUOOIXDpZn5k03FB0AgsEX?usp=sharing

## Results
* train wav2vec and linear classifier on denoised audio:
  ![epoch-all-denoised](https://github.com/user-attachments/assets/d754de0d-c29a-4c5f-b332-162bd9d9483d)
* train linear classifider on denoised audio:
  ![epoch-denoised](https://github.com/user-attachments/assets/494be666-60ec-493f-b5a9-bfb903f6f853)
* train linear classifier on noisy audio:
  ![epoch-noisy](https://github.com/user-attachments/assets/df560fe1-3296-4a5c-95f3-aff938eaa425)


## Features

- fine-tunes classifier-head for Wav2Vec2 model for binary classification (speech/non-speech)
- processes audio in fixed-length windows with configurable stride
- handles class imbalance using weighted loss
- training visualization with loss and F1 curves

## Files

- `train.py`: main training script with model setup, training loop, and evaluation
- `VAD_dataset.py`: custom dataset class for loading and preprocessing audio data
- `test.json` and `dev.json`: dummy manifest files for testing purposes


## Run training:
```bash
python train.py > output.txt
