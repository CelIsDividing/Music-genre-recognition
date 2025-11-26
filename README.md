# Klasifikacija Muzičkih Žanrova pomoću Deep Learning-a

## Opis projekta
Projektat koristi različite neuronske mreže za klasifikaciju muzičkih žanrova na osnovu audio spektrograma. Uključuje poređenje više modela: MLP, jednostavne CNN, napredne CNN i ResNet arhitekture.

## Podržani žanrovi
Projekt podržava 10 muzičkih žanrova:
- Country
- Jazz
- Pop
- Rap
- Rock
- Classical
- Blues
- Electronic
- Metal
- Folk

## Instalacija
```bash
pip install -r requirements.txt
```

## Priprema podataka
1. Postavite audio fajlove (.wav ili .mp3) u odgovarajuće foldere u `data/raw/` po žanrovima
2. Svaki žanr treba da ima bar 100 audio uzoraka za dobar trening

**Napomena:** Za automatsko preuzimanje i organizovanje audio uzoraka možete koristiti `download_data` klasu iz `scripts/download_data.py`. Ova klasa vam pomaže da importujete svoje audio uzorke i organizujete ih u odgovarajuću strukturu foldera.

## Pokretanje

### Trening pojedinačnog modela
```bash
python scripts/train_model.py
```

### Poređenje više modela
```bash
python scripts/compare_models.py
```
Ovaj skript poredi 4 različita modela i generiše tabelu sa rezultatima u `results/model_comparison.csv`

### Predikcija žanra za novi audio
```bash
python scripts/predict.py path/to/audio_file.wav
```

### Evaluacija modela
```bash
python scripts/evaluate.py
```

## Struktura projekta
- `data/raw/` - Neobrađeni audio fajlovi po žanrovima
- `data/processed/` - Obradjeni spektrogrami i karakteristike
- `models/` - Sačuvani modeli i encoderi
- `src/` - Izvorni kod (feature extraction, model, trening, predikcija)
- `scripts/` - Skripte za pokretanje
- `results/` - Rezultati i grafikoni

## Ključne komponente
- **Ekstrakcija karakteristika:** Mel-spektrogrami iz audio fajlova
- **Modeli:** MLP, jednostavna CNN, napredna CNN, ResNet
- **Preprocesiranje:** Normalizacija i reshaping podataka
- **Obučavanje:** Early Stopping i Learning Rate Scheduling
- **Evaluacija:** Confusion matrix, classification report i poređenje modela

## Zahtevi
- Python 3.8+
- TensorFlow, librosa, scikit-learn, matplotlib, seaborn, pandas
