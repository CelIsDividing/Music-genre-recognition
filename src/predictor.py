import numpy as np

class Predictor:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def predict_genre(self, spectrogram):
        pred = self.model.predict(spectrogram)
        genre_idx = np.argmax(pred, axis=1)[0]
        genre = self.label_encoder.inverse_transform([genre_idx])[0]
        return genre
