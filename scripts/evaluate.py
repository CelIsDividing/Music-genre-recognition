import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from keras.models import load_model
from src.data_loader import DataLoader
from src.utils import plot_confusion_matrix, print_classification_report

# Putanje
model_path = 'models/music_genre_classifier.h5'
encoder_path = 'models/label_encoder.pkl'
data_dir = 'data/raw'

# Učitaj model i encoder
model = load_model(model_path)
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Učitaj podatke
loader = DataLoader({})
features, labels = loader.load_audio_files(data_dir)
features = features[..., None]

# Predikcija
y_pred = model.predict(features)
y_pred = y_pred.argmax(axis=1)
labels_str = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

# Evaluacija
plot_confusion_matrix(labels, y_pred, labels_str, save_path='results/confusion_matrix.png')
report = print_classification_report(labels, y_pred, labels_str)
with open('results/performance_report.txt', 'w') as f:
    f.write(report)
