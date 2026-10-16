from colorama import Fore, Style, init
init(autoreset=True)
import sys
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow logs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_extractor import FeatureExtractor
from src.predictor import Predictor
import numpy as np
from keras.models import load_model
import pickle

# Paths
model_path = 'models/music_genre_classifier.h5'
encoder_path = 'models/label_encoder.pkl'
audio_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/classical/Air - Johann Sebastian Bach.wav'

# Check whether the file exists
if not os.path.exists(audio_path):
    print(f"{Fore.RED}Error: Audio file '{audio_path}' does not exist!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Usage: python scripts/predict.py <path_to_audio_file>{Style.RESET_ALL}")
    sys.exit(1)

# Load model and encoder
try:
    model = load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    print(f"{Fore.RED}Error: Model or label encoder not found! Run training first.{Style.RESET_ALL}")
    sys.exit(1)

extractor = FeatureExtractor()
spectrogram = extractor.extract_mel_spectrogram(audio_path)
spectrogram = np.squeeze(spectrogram)
if spectrogram.ndim == 2:
    spectrogram = np.expand_dims(spectrogram, axis=-1)
spectrogram = np.expand_dims(spectrogram, axis=0)

predictor = Predictor(model, label_encoder)
predicted_genre = predictor.predict_genre(spectrogram)

# Nicer user output
print(f"{Fore.MAGENTA}{'-'*50}")
print(f"{Fore.WHITE}Your song has been analyzed and belongs to the genre:")
print(f"{Fore.GREEN}{Style.BRIGHT}>>> {Fore.YELLOW}{predicted_genre.upper()} {Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'-'*50}")
print(f"{Fore.CYAN}{'-'*40}")
