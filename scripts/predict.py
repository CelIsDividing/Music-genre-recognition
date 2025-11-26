from colorama import Fore, Style, init
init(autoreset=True)
import sys
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Sakrij TensorFlow logove
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_extractor import FeatureExtractor
from src.predictor import Predictor
import numpy as np
from keras.models import load_model
import pickle

# Putanje
model_path = 'models/music_genre_classifier.h5'
encoder_path = 'models/label_encoder.pkl'
audio_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/classical/Air - Johann Sebastian Bach.wav'

# Proveri da li fajl postoji
if not os.path.exists(audio_path):
    print(f"{Fore.RED}❌ Greška: Audio fajl '{audio_path}' ne postoji!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}💡 Koristite: python scripts/predict.py <putanja_do_audio_fajla>{Style.RESET_ALL}")
    sys.exit(1)

# Učitaj model i encoder
try:
    model = load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    print(f"{Fore.RED}❌ Greška: Model ili label encoder nisu pronađeni! Prvo pokrenite trening.{Style.RESET_ALL}")
    sys.exit(1)

extractor = FeatureExtractor()
spectrogram = extractor.extract_mel_spectrogram(audio_path)
spectrogram = np.squeeze(spectrogram)
if spectrogram.ndim == 2:
    spectrogram = np.expand_dims(spectrogram, axis=-1)
spectrogram = np.expand_dims(spectrogram, axis=0)

predictor = Predictor(model, label_encoder)
predicted_genre = predictor.predict_genre(spectrogram)

# Lepši korisnički output
print(f"{Fore.MAGENTA}{'-'*50}")
print(f"{Fore.WHITE}Vaša pesma je analizirana i pripada žanru:")
print(f"{Fore.GREEN}{Style.BRIGHT}>>> {Fore.YELLOW}{predicted_genre.upper()} {Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'-'*50}")
print(f"{Fore.CYAN}{'-'*40}")
