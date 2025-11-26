import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, X_train, y_train, X_val, y_val):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get(
                'learning_rate', 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.get('epochs', 50),
            batch_size=self.config.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        return history
