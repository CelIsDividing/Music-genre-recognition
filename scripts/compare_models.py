import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from colorama import Fore, Style, init

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.model import GenreClassifier
from src.trainer import Trainer

init(autoreset=True)

# Configuration
config = {
    'learning_rate': 0.001,
    'epochs': 50, 
    'batch_size': 8
}

# List of models to compare
models_to_compare = [
    ('MLP', 'build_mlp_model'),
    ('Simple CNN', 'build_simple_cnn_model'),
    ('Advanced CNN', 'build_cnn_model'),
    ('ResNet', 'build_resnet_model')
]

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns metrics"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    print(f"{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}MODEL COMPARISON FOR MUSIC GENRE CLASSIFICATION{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}")

    # Load data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    print(f"{Fore.BLUE}{Style.BRIGHT}Loading audio files...{Style.RESET_ALL}")
    loader = DataLoader(config)
    features, labels = loader.load_audio_files(data_dir)
    print(f"{Fore.GREEN}Audio files loaded! Found {len(features)} samples.{Style.RESET_ALL}")

    # Reshape for CNN
    features = features[..., None]

    # Split data
    print(f"{Fore.BLUE}{Style.BRIGHT}Splitting data...{Style.RESET_ALL}")
    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    print(f"{Fore.GREEN}Data split: Training {len(X_train)}, Validation {len(X_val)}, Test {len(X_test)}{Style.RESET_ALL}")

    num_classes = len(set(labels))
    input_shape = X_train.shape[1:]

    results = []

    for model_name, model_method in models_to_compare:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}Training model: {model_name}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'-'*40}{Style.RESET_ALL}")

        # Create model
        classifier = GenreClassifier()
        model = getattr(classifier, model_method)(input_shape, num_classes)

        # Training
        trainer = Trainer(model, config)
        history = trainer.train(X_train, y_train, X_val, y_val)

        # Evaluation
        print(f"{Fore.BLUE}Evaluating on the test set...{Style.RESET_ALL}")
        metrics = evaluate_model(model, X_test, y_test)

        # Save model
        model_path = f'models/{model_name.lower().replace(" ", "_")}_model.h5'
        model.save(model_path)

        # Add results
        result = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'Val_Accuracy': max(history.history['val_accuracy']),
            'Val_Loss': min(history.history['val_loss'])
        }
        results.append(result)

        print(f"{Fore.GREEN}{model_name} finished! Accuracy: {metrics['accuracy']:.4f}{Style.RESET_ALL}")

    # Compare results
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}COMPARISON RESULTS{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}")

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False, float_format='%.4f'))

    # Save results
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/model_comparison.csv', index=False)

    # Best model
    best_model = df_results.loc[df_results['Accuracy'].idxmax()]
    print(f"\n{Fore.GREEN}{Style.BRIGHT}BEST MODEL: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f}){Style.RESET_ALL}")

    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(loader.label_encoder, f)

    print(f"\n{Fore.CYAN}All results saved to results/model_comparison.csv{Style.RESET_ALL}")

if __name__ == "__main__":
    main()