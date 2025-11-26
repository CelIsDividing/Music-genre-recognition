import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=22050, duration=20, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
    
    def extract_mel_spectrogram(self, audio_path):
        """Ekstrahuje Mel-spektrogram iz audio fajla"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        # Normalizacija na [0, 1]
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        return S_db_norm
    
    def extract_mfcc(self, audio_path, n_mfcc=13):
        """Ekstrahuje MFCC (Mel-Frequency Cepstral Coefficients) karakteristike
        
        MFCC su široko korišćene karakteristike u audio obradi koje modeluju
        kako ljudsko uho percipira zvuk. Koriste se za:
        - Prepoznavanje govora
        - Klasifikaciju muzičkih žanrova
        - Analizu audio signala
        
        Args:
            audio_path (str): Putanja do audio fajla
            n_mfcc (int): Broj MFCC koeficijenata (tipično 13-40)
            
        Returns:
            np.ndarray: MFCC matrice (n_mfcc x vreme)
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Normalizacija po vremenskim frejmovima
        mfcc_norm = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        return mfcc_norm