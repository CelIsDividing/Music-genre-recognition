# Seminar: Klasifikacija Muzičkih Žanrova pomoću Deep Learning-a

## Uvod

U savremenom svetu, muzika igra značajnu ulogu u svakodnevnom životu ljudi. Sa ogromnom količinom dostupnih muzičkih sadržaja na platformama kao što su Spotify, YouTube i Apple Music, postaje sve važnije automatski klasifikovati muzičke žanrove. Ova klasifikacija pomaže u organizovanju muzičkih biblioteka, preporučivanju sadržaja korisnicima i analizi muzičkih trendova.

Ovaj seminar predstavlja projekat koji koristi duboko učenje za klasifikaciju muzičkih žanrova na osnovu audio spektrograma. Projekat implementira i poredi četiri različita modela neuronskih mreža: Višeslojnu neuronsku mrežu (MLP), jednostavnu konvolucionu neuronsku mrežu (CNN), naprednu CNN i ResNet arhitekturu. Cilj je demonstrirati kako se mašinsko učenje može primeniti na audio podatke za rešavanje problema klasifikacije.

## Opis problema

Klasifikacija muzičkih žanrova predstavlja izazovan problem u oblasti mašinskog učenja zbog kompleksnosti audio signala. Audio fajlovi sadrže veliku količinu podataka koji se menjaju tokom vremena, što zahteva efikasne metode ekstrakcije karakteristika. Tradicionalni pristupi zasnovani na ručno izabranim karakteristikama (kao što su MFCC, spektrogrami ili tempo) često nisu dovoljno robusni za različite vrste muzike.

Problem se može formulisati kao višeklasna klasifikacija gde je ulaz audio signal, a izlaz jedan od 10 muzičkih žanrova: Country, Jazz, Pop, Rap, Rock, Classical, Blues, Electronic, Metal i Folk. Svaki žanr ima svoje karakteristične audio osobine koje modeli dubokog učenja mogu naučiti da prepoznaju.

Glavni izazovi uključuju:
- Veliku dimenzionalnost audio podataka
- Varijabilnost unutar žanrova
- Potrebu za velikim skupovima podataka za trening
- Računarsku kompleksnost dubokih modela

## Način rada i alati

Za realizaciju ovog projekta koristi se programski jezik Python zbog svoje popularnosti u oblasti mašinskog učenja i bogatog ekosistema biblioteka. Kao glavni okvir za duboko učenje koristi se Keras, API koji radi iznad TensorFlow-a. Keras omogućava jednostavno definisanje i trening neuronskih mreža.

Ključne biblioteke koje se koriste:

- **TensorFlow/Keras**: Za izgradnju i trening neuronskih mreža
- **Librosa**: Za obradu audio signala i ekstrakciju karakteristika
- **NumPy**: Za numeričke operacije i rad sa nizovima
- **Pandas**: Za manipulaciju podacima i analizu rezultata
- **Scikit-learn**: Za podelu podataka i evaluacione metrike
- **Matplotlib/Seaborn**: Za vizualizaciju rezultata

Proces rada se sastoji iz nekoliko koraka:
1. Prikupljanje i priprema audio podataka
2. Ekstrakcija karakteristika (Mel-spektrogrami)
3. Podela podataka na trening, validaciju i test skupove
4. Definisanje i trening različitih modela
5. Evaluacija performansi modela
6. Poređenje rezultata

## Kako radi model

Modeli koriste Mel-spektrograme kao ulazne karakteristike. Mel-spektrogram je dvodimenzionalna reprezentacija audio signala koja prikazuje kako se energija signala menja tokom vremena u različitim frekvencijskim opsezima. Ova reprezentacija je posebno pogodna za muzičku analizu jer odražava kako ljudsko uho percipira zvuk.

### Višeslojna neuronska mreža (MLP)
MLP je najjednostavniji model koji se sastoji od potpuno povezanih slojeva. Ulazni spektrogram se prvo spljošti u vektor, zatim prolazi kroz nekoliko gustih slojeva sa aktivacijskim funkcijama ReLU. Koriste se Batch Normalization i Dropout slojevi za regularizaciju.

### Jednostavna CNN
Konvoluciona neuronska mreža koristi konvolucione slojeve za ekstrakciju prostornih karakteristika iz spektrograma. Model ima dva konvoluciona bloka sa MaxPooling slojevima za smanjenje dimenzionalnosti.

### Napredna CNN
Ovaj model ima više konvolucionih slojeva (32, 64, 128 filtera) sa Batch Normalization i Dropout slojevima između njih. Koristi se za ekstrakciju složenijih karakteristika.

### ResNet
ResNet arhitektura koristi residualne blokove koji omogućavaju trening veoma dubokih mreža. Svaki residualni blok sadrži dva konvoluciona sloja sa preskočnim vezama koje pomažu u rešavanju problema nestajućeg gradijenta.

Svi modeli koriste Adam optimizator, sparse categorical cross-entropy loss funkciju i accuracy kao metriku. Trening uključuje Early Stopping i Learning Rate Scheduling za sprečavanje preobučavanja.

## Praktični deo tj. kod

### Ekstrakcija karakteristika
```python
import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=22050, duration=20, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
    
    def extract_mel_spectrogram(self, audio_path):
        """Ekstrahuje Mel-spektrogram iz audio fajla"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        # Normalizacija na [0, 1]
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        return S_db_norm
```

### Definisanje modela
```python
from keras import layers, models

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
```

### Trening modela
```python
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
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
```

## Rezultat koda

Nakon treninga i evaluacije četiri modela na test skupu, dobijeni su sledeći rezultati:

| Model       | Accuracy | Precision | Recall | F1-Score | Val_Accuracy | Val_Loss |
|-------------|----------|-----------|--------|----------|--------------|----------|
| MLP         | 0.2000   | 0.1374    | 0.2000 | 0.1350   | 0.2941       | 2.1571   |
| Simple CNN  | 0.2235   | 0.2549    | 0.2235 | 0.2099   | 0.2794       | 1.9989   |
| Advanced CNN| 0.4235   | 0.4783    | 0.4235 | 0.3971   | 0.3824       | 3.0516   |
| ResNet      | 0.3647   | 0.3087    | 0.3647 | 0.2770   | 0.2500       | 2.0949   |

Najbolji rezultati postignuti su sa naprednom CNN arhitekturom koja je ostvarila tačnost od 42.35% na test skupu. Ova tačnost je relativno niska zbog ograničenog skupa podataka i kompleksnosti problema klasifikacije muzičkih žanrova. MLP i jednostavna CNN su pokazali slične performanse (~20-22%), dok je ResNet bio nešto bolji od MLP-a ali lošiji od napredne CNN.

Validacione metrike pokazuju da modeli nisu preobučeni, što je dobro. Međutim, niske vrednosti tačnosti ukazuju na potrebu za većim skupovima podataka i dodatnim tehnikama augmentacije.

## Zaključak

Ovaj projekat je demonstrirao primenu dubokog učenja na problem klasifikacije muzičkih žanrova. Iako su rezultati ograničeni veličinom skupa podataka, projekat pokazuje važnost različitih arhitektura neuronskih mreža za audio klasifikaciju.

Ključni uvidi:
- Konvolucione neuronske mreže su superiorne u odnosu na MLP za obradu spektrograma
- Naprednija CNN arhitektura daje bolje rezultate od jednostavne
- ResNet pokazuje obećavajuće rezultate ali zahteva više podataka za optimalne performanse

Za budući rad, preporučuje se:
- Prikupljanje većeg skupa podataka
- Implementacija data augmentation tehnika
- Istraživanje drugih karakteristika (MFCC, chroma, tempo)
- Upotreba transfer learning-a sa pretreniranim modelima
- Optimizacija hiperparametara

Ovaj projekat predstavlja solidnu osnovu za dalji razvoj sistema za automatsku klasifikaciju muzičkih žanrova.