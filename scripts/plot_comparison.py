import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
results_path = 'results/model_comparison.csv'
df = pd.read_csv(results_path)

# Create a bar chart for accuracy
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['Accuracy'], color=['skyblue', 'lightgreen', 'orange', 'red'])

# Add values on the bars
for bar, accuracy in zip(bars, df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold')

plt.title('Model accuracy comparison for music genre classification', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 0.5)  # Set the y-axis to 50% because the values are low
plt.grid(axis='y', alpha=0.3)

# Save the chart
plt.tight_layout()
plt.savefig('results/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved to results/model_accuracy_comparison.png")

# Display the chart (if possible)
plt.show()