from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.model import GenreClassifier
from src.trainer import ModelTrainer
from src.config import CONFIG

def main():
    # Initialize components
    data_loader = DataLoader(CONFIG)
    feature_extractor = FeatureExtractor(CONFIG['audio'])
    model = GenreClassifier()
    trainer = ModelTrainer(CONFIG['model'])
    
    # Load data
    print("Loading data...")
    audio_data, labels = data_loader.load_audio_files(CONFIG['paths']['data_dir'])
    
    # Extract features
    print("Extracting features...")
    features = feature_extractor.extract_features(audio_data)
    
    # Train the model
    print("Training the model...")
    trained_model, history = trainer.train_model(features, labels, model)
    
    # Save the model
    trainer.save_model(trained_model, CONFIG['paths']['model_dir'])
    
    print("Project completed successfully!")

if __name__ == "__main__":
    main()