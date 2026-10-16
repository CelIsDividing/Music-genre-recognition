#!/usr/bin/env python3
"""
Simple runner for the entire project
"""

import sys
import os

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from music_genre_classifier import MusicGenreClassifier

def run_complete_project():
    classifier = MusicGenreClassifier()
    
    # Check whether the dataset exists
    if not os.path.exists('data/raw'):
        print("Dataset does not exist! Create the data/raw directory with genres.")
        print("Structure: data/raw/pop/, data/raw/rock/, etc.")
        return
    
    # Run the complete pipeline
    classifier.run_full_pipeline()

if __name__ == "__main__":
    run_complete_project()