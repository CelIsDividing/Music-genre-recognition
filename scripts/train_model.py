import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Sakrij TensorFlow logove
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import DataLoader
from src.model import GenreClassifier
from src.trainer import Trainer
from colorama import Fore, Style, init
init(autoreset=True)

# Konfiguracija
config = {
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 8
}

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
print(f"{Fore.BLUE}{Style.BRIGHT}🔊 Učitavanje audio fajlova...{Style.RESET_ALL}")
loader = DataLoader(config)
features, labels = loader.load_audio_files(data_dir)
print(f"{Fore.GREEN}✅ Audio fajlovi uspešno učitani! Pronađeno {len(features)} uzoraka.{Style.RESET_ALL}")

# Reshape za CNN (dodaj kanal dimenziju)
features = features[..., None]

# Split na train/val/test
print(f"{Fore.BLUE}{Style.BRIGHT}📊 Deljenje podataka na trening, validaciju i test skup...{Style.RESET_ALL}")
X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
print(f"{Fore.GREEN}✅ Podaci podeljeni: Trening {len(X_train)}, Validacija {len(X_val)}, Test {len(X_test)}{Style.RESET_ALL}")

# Model
num_classes = len(set(labels))
input_shape = X_train.shape[1:]
print(f"{Fore.BLUE}{Style.BRIGHT}🏗️  Izgradnja CNN modela...{Style.RESET_ALL}")
classifier = GenreClassifier()
model = classifier.build_cnn_model(input_shape, num_classes)
print(f"{Fore.GREEN}✅ Model uspešno izgrađen! Broj klasa: {num_classes}{Style.RESET_ALL}")

# Trening
print(f"{Fore.BLUE}{Style.BRIGHT}🚀 Pokretanje treninga...{Style.RESET_ALL}")
trainer = Trainer(model, config)
history = trainer.train(X_train, y_train, X_val, y_val)

# Sačuvaj model
print(f"{Fore.BLUE}{Style.BRIGHT}💾 Čuvanje modela...{Style.RESET_ALL}")
model.save('models/music_genre_classifier.h5')

# Sačuvaj label encoder
import pickle
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(loader.label_encoder, f)

print(f"{Fore.GREEN}✅ Model i label encoder uspešno sačuvani!{Style.RESET_ALL}")

# Lepši korisnički output
print(f"{Fore.MAGENTA}{'='*50}")
print(f"{Fore.CYAN}{Style.BRIGHT}🎉 Trening potpuno završen! 🎉{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'-'*50}")
print(f"{Fore.WHITE}📁 Model sačuvan kao: {Fore.YELLOW}models/music_genre_classifier.h5{Style.RESET_ALL}")
print(f"{Fore.WHITE}📁 Label encoder sačuvan kao: {Fore.YELLOW}models/label_encoder.pkl{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'-'*50}")
