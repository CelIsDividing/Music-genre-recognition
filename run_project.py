#!/usr/bin/env python3
"""
Jednostavno pokretanje celog projekta
"""

import sys
import os

# Dodavanje src putanje
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from music_genre_classifier import MusicGenreClassifier

def run_complete_project():
    classifier = MusicGenreClassifier()
    
    # Provera da li dataset postoji
    if not os.path.exists('data/raw'):
        print("Dataset ne postoji! Kreirajte direktorijum data/raw sa žanrovima.")
        print("Struktura: data/raw/pop/, data/raw/rock/, itd.")
        return
    
    # Pokretanje celokupnog pipeline-a
    classifier.run_full_pipeline()

if __name__ == "__main__":
    run_complete_project()