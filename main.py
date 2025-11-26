from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.model import GenreClassifier
from src.trainer import ModelTrainer
from src.config import CONFIG

def main():
    # Inicijalizacija komponenti
    data_loader = DataLoader(CONFIG)
    feature_extractor = FeatureExtractor(CONFIG['audio'])
    model = GenreClassifier()
    trainer = ModelTrainer(CONFIG['model'])
    
    # Učitavanje podataka
    print("Učitavanje podataka...")
    audio_data, labels = data_loader.load_audio_files(CONFIG['paths']['data_dir'])
    
    # Ekstrakcija karakteristika
    print("Ekstrakcija karakteristika...")
    features = feature_extractor.extract_features(audio_data)
    
    # Obučavanje modela
    print("Obučavanje modela...")
    trained_model, history = trainer.train_model(features, labels, model)
    
    # Čuvanje modela
    trainer.save_model(trained_model, CONFIG['paths']['model_dir'])
    
    print("Projekat uspešno završen!")

if __name__ == "__main__":
    main()