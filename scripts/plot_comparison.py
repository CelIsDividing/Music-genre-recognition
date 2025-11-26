import pandas as pd
import matplotlib.pyplot as plt
import os

# Učitaj rezultate
results_path = 'results/model_comparison.csv'
df = pd.read_csv(results_path)

# Kreiraj bar chart za tačnost
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['Accuracy'], color=['skyblue', 'lightgreen', 'orange', 'red'])

# Dodaj vrednosti na barovima
for bar, accuracy in zip(bars, df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold')

plt.title('Poređenje tačnosti modela za klasifikaciju muzičkih žanrova', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Tačnost (Accuracy)', fontsize=12)
plt.ylim(0, 0.5)  # Postavi y-osu do 50% jer su vrednosti niske
plt.grid(axis='y', alpha=0.3)

# Sačuvaj grafikon
plt.tight_layout()
plt.savefig('results/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("📊 Grafikon sačuvan u results/model_accuracy_comparison.png")

# Prikaži grafikon (ako je moguće)
plt.show()