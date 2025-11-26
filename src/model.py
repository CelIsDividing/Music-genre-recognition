from keras import layers, models
import tensorflow as tf

class GenreClassifier:
    def build_mlp_model(self, input_shape, num_classes):
        """Gradi višeslojnu neuronsku mrežu (MLP) za klasifikaciju žanrova"""
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def build_simple_cnn_model(self, input_shape, num_classes):
        """Gradi jednostavnu CNN mrežu sa manje slojeva"""
        model = models.Sequential([
            layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.2),

            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.2),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def build_cnn_model(self, input_shape, num_classes):
        """Gradi CNN model za klasifikaciju žanrova na osnovu spektrograma"""
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def build_resnet_model(self, input_shape, num_classes):
        """Gradi ResNet model za klasifikaciju žanrova"""
        inputs = layers.Input(shape=input_shape)

        # Prvi konvolucioni blok
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

        # Residual blokovi
        x = self._resnet_block(x, 64, 3)
        x = self._resnet_block(x, 128, 4, strides=2)
        x = self._resnet_block(x, 256, 6, strides=2)
        x = self._resnet_block(x, 512, 3, strides=2)

        # Global average pooling i dense
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        return model

    def _resnet_block(self, x, filters, blocks, strides=1):
        """Pomoćna funkcija za ResNet blok"""
        for i in range(blocks):
            shortcut = x
            if i == 0 and strides > 1:
                shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)

            x = layers.Conv2D(filters, (3, 3), strides=strides if i == 0 else 1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)

        return x