import matplotlib.pyplot as plt
import pandas as pd

def compare_results(classical_acc, quantum_acc):
    df = pd.DataFrame({
        'Model': ['Classical', 'Quantum'],
        'Accuracy': [classical_acc, quantum_acc]
    })
    df.to_csv('results/accuracy_comparison.csv', index=False)
    print("\nResults saved to results/accuracy_comparison.csv")

def plot_comparison(classical_acc, quantum_acc):
    plt.bar(['Classical', 'Quantum'], [classical_acc, quantum_acc], color=['skyblue', 'purple'])
    plt.title('Accuracy Comparison: Classical vs Quantum')
    plt.ylabel('Accuracy')
    plt.savefig('results/charts/comparison.png')
    plt.show()
