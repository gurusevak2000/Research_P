"""
Exploring Quantum Machine Learning:
A Comparative Study of Classical vs Quantum Algorithms
Author: [Your Name]
MCA Research Project
"""

from src.classical_model import run_classical_model
from src.quantum_model import run_quantum_model
from src.utils import compare_results, plot_comparison

def main():
    print("Starting Classical Model...")
    classical_accuracy = run_classical_model()

    print("\nStarting Quantum Model Simulation...")
    quantum_accuracy = run_quantum_model()

    print("\nComparing Results...")
    compare_results(classical_accuracy, quantum_accuracy)
    plot_comparison(classical_accuracy, quantum_accuracy)

if __name__ == "__main__":
    main()
