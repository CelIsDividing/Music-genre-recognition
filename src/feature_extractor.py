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
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        return S_db_norm
    
    def extract_mfcc(self, audio_path, n_mfcc=13):
        """Extracts MFCC (Mel-Frequency Cepstral Coefficients) features
        
        MFCCs are widely used features in audio processing that model
        how the human ear perceives sound. They are used for:
        - Speech recognition
        - Music genre classification
        - Audio signal analysis
        
        Args:
            audio_path (str): Path to the audio file
            n_mfcc (int): Number of MFCC coefficients (typically 13-40)
            
        Returns:
            np.ndarray: MFCC matrix (n_mfcc x time)
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Normalization across time frames
        mfcc_norm = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        return mfcc_norm