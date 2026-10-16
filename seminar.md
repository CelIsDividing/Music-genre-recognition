# Seminar: Music Genre Classification with Deep Learning

## Introduction

In the modern world, music plays a significant role in people's everyday lives. With the huge amount of music content available on platforms such as Spotify, YouTube and Apple Music, it is becoming increasingly important to automatically classify music genres. This classification helps to organize music libraries, recommend content to users and analyze music trends.

This seminar presents a project that uses deep learning to classify music genres based on audio spectrograms. The project implements and compares four different neural network models: a multilayer perceptron (MLP), a simple convolutional neural network (CNN), an advanced CNN and a ResNet architecture. The goal is to demonstrate how machine learning can be applied to audio data to solve a classification problem.

## Problem description

Music genre classification is a challenging problem in the field of machine learning due to the complexity of audio signals. Audio files contain a large amount of data that changes over time, which requires efficient feature extraction methods. Traditional approaches based on hand-picked features (such as MFCC, spectrograms or tempo) are often not robust enough for different types of music.

The problem can be formulated as multiclass classification where the input is an audio signal and the output is one of 10 music genres: Country, Jazz, Pop, Rap, Rock, Classical, Blues, Electronic, Metal and Folk. Each genre has its own characteristic audio properties that deep learning models can learn to recognize.

The main challenges include:
- The high dimensionality of audio data
- Variability within genres
- The need for large training datasets
- The computational complexity of deep models

## Approach and tools

To implement this project, the Python programming language is used because of its popularity in the field of machine learning and its rich ecosystem of libraries. Keras, an API that runs on top of TensorFlow, is used as the main deep learning framework. Keras makes it easy to define and train neural networks.

The key libraries used:

- **TensorFlow/Keras**: For building and training neural networks
- **Librosa**: For audio signal processing and feature extraction
- **NumPy**: For numerical operations and working with arrays
- **Pandas**: For data manipulation and results analysis
- **Scikit-learn**: For data splitting and evaluation metrics
- **Matplotlib/Seaborn**: For results visualization

The workflow consists of several steps:
1. Collecting and preparing audio data
2. Feature extraction (Mel-spectrograms)
3. Splitting data into training, validation and test sets
4. Defining and training different models
5. Evaluating model performance
6. Comparing the results

## How the model works

The models use Mel-spectrograms as input features. A Mel-spectrogram is a two-dimensional representation of an audio signal that shows how the signal's energy changes over time across different frequency ranges. This representation is particularly suitable for music analysis because it reflects how the human ear perceives sound.

### Multilayer perceptron (MLP)
The MLP is the simplest model, consisting of fully connected layers. The input spectrogram is first flattened into a vector, then passes through several dense layers with ReLU activation functions. Batch Normalization and Dropout layers are used for regularization.

### Simple CNN
A convolutional neural network uses convolutional layers to extract spatial features from the spectrogram. The model has two convolutional blocks with MaxPooling layers to reduce dimensionality.

### Advanced CNN
This model has more convolutional layers (32, 64, 128 filters) with Batch Normalization and Dropout layers between them. It is used to extract more complex features.

### ResNet
The ResNet architecture uses residual blocks that enable the training of very deep networks. Each residual block contains two convolutional layers with skip connections that help solve the vanishing gradient problem.

All models use the Adam optimizer, the sparse categorical cross-entropy loss function and accuracy as the metric. Training includes Early Stopping and Learning Rate Scheduling to prevent overfitting.

## Practical part, i.e. the code

### Feature extraction
```python
import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=22050, duration=20, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
    
    def extract_mel_spectrogram(self, audio_path):
        """Extracts a Mel-spectrogram from an audio file"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        # Normalization to [0, 1]
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        return S_db_norm
```

### Model definition
```python
from keras import layers, models

class GenreClassifier:
    def build_mlp_model(self, input_shape, num_classes):
        """Builds a multilayer perceptron (MLP) for genre classification"""
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def build_simple_cnn_model(self, input_shape, num_classes):
        """Builds a simple CNN with fewer layers"""
        model = models.Sequential([
            layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
```

### Model training
```python
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, X_train, y_train, X_val, y_val):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.get('epochs', 50),
            batch_size=self.config.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        return history
```

## Code results

After training and evaluating the four models on the test set, the following results were obtained:

| Model       | Accuracy | Precision | Recall | F1-Score | Val_Accuracy | Val_Loss |
|-------------|----------|-----------|--------|----------|--------------|----------|
| MLP         | 0.2000   | 0.1374    | 0.2000 | 0.1350   | 0.2941       | 2.1571   |
| Simple CNN  | 0.2235   | 0.2549    | 0.2235 | 0.2099   | 0.2794       | 1.9989   |
| Advanced CNN| 0.4235   | 0.4783    | 0.4235 | 0.3971   | 0.3824       | 3.0516   |
| ResNet      | 0.3647   | 0.3087    | 0.3647 | 0.2770   | 0.2500       | 2.0949   |

The best results were achieved with the advanced CNN architecture, which reached an accuracy of 42.35% on the test set. This accuracy is relatively low due to the limited dataset and the complexity of the music genre classification problem. The MLP and the simple CNN showed similar performance (~20-22%), while ResNet was somewhat better than the MLP but worse than the advanced CNN.

The validation metrics show that the models are not overfitted, which is good. However, the low accuracy values indicate a need for larger datasets and additional augmentation techniques.

## Conclusion

This project demonstrated the application of deep learning to the problem of music genre classification. Although the results are limited by the size of the dataset, the project shows the importance of different neural network architectures for audio classification.

Key insights:
- Convolutional neural networks are superior to the MLP for processing spectrograms
- A more advanced CNN architecture gives better results than a simple one
- ResNet shows promising results but requires more data for optimal performance

For future work, the following is recommended:
- Collecting a larger dataset
- Implementing data augmentation techniques
- Exploring other features (MFCC, chroma, tempo)
- Using transfer learning with pretrained models
- Hyperparameter optimization

This project represents a solid foundation for further development of an automatic music genre classification system.
