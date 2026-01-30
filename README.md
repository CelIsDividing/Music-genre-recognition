# Music Genre Classification using Deep Learning

## Project description
This project uses various neural networks to classify music genres based on audio spectrograms. It includes a comparison of multiple models: MLP, simple CNNs, advanced CNNs and ResNet architectures.

## Supported genres
The project supports 10 music genres:
- Country
- Jazz
- Pop
- Rap
- Rock
- Classical
- Blues
- Electronic
- Metal
- Folk

## Installation
```bash
pip install -r requirements.txt
```

## Data preparation
1. Place audio files (.wav or .mp3) into the appropriate folders under `data/raw/`, organized by genre.
2. Each genre should have at least 100 audio samples for good training.

**Note:** For automatic downloading and organizing of audio samples you can use the `download_data` class from `scripts/download_data.py`. This class helps you import and organize your audio samples automatically.

## Running

### Training a single model
```bash
python scripts/train_model.py
```

### Comparing multiple models
```bash
python scripts/compare_models.py
```
This script compares 4 different models and generates a results table at `results/model_comparison.csv`.

### Predicting the genre of a new audio file
```bash
python scripts/predict.py path/to/audio_file.wav
```

### Evaluating a model
```bash
python scripts/evaluate.py
```

## Project structure
- `data/raw/` - Raw audio files organized by genre
- `data/processed/` - Processed spectrograms and features
- `models/` - Saved models and encoders
- `src/` - Source code (feature extraction, models, training, prediction)
- `scripts/` - Scripts for running tasks
- `results/` - Results and plots

## Key components
- **Feature extraction:** Mel-spectrograms from audio files
- **Models:** MLP, simple CNN, advanced CNN, ResNet
- **Preprocessing:** Normalization and reshaping of data
- **Training:** Early stopping and learning rate scheduling
- **Evaluation:** Confusion matrix, classification report and model comparisons

## Requirements
- Python 3.8+
- TensorFlow, librosa, scikit-learn, matplotlib, seaborn, pandas
