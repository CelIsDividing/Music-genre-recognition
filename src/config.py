# Configuration parameters
CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'duration': 30,
        'n_mels': 128,
        'hop_length': 512
    },
    'model': {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001
    },
    'paths': {
        'data_dir': 'data/raw',
        'model_dir': 'models',
        'results_dir': 'results'
    },
    'genres': ['pop', 'rock', 'jazz', 'classical', 'hiphop', 
               'electronic', 'blues', 'country', 'electronic', 'folk']
}