import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
    
    def load_audio_files(self, data_dir):
        """Učitava audio fajlove iz direktorijuma po žanrovima"""
        from src.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        features = []
        labels = []
        total_files = 0
        loaded_files = 0
        for genre in os.listdir(data_dir):
            genre_path = os.path.join(data_dir, genre)
            if not os.path.isdir(genre_path):
                continue
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    total_files += 1
                    file_path = os.path.join(genre_path, file)
                    try:
                        mel_spec = extractor.extract_mel_spectrogram(file_path)
                        features.append(mel_spec)
                        labels.append(genre)
                        loaded_files += 1
                    except Exception as e:
                        print(f"⚠️  Preskačen problematični fajl: {file_path} - Greška: {str(e)}")
        print(f"📊 Ukupno fajlova: {total_files}, Uspešno učitano: {loaded_files}")
        features = np.array(features)
        labels = self.label_encoder.fit_transform(labels)
        return features, labels
    
    def split_dataset(self, features, labels):
        """Deli dataset na trening, validacioni i test skup"""
        # Implementacija...