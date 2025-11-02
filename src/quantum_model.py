from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np

def run_quantum_model():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X, y = X[y != 2], y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize features between 0 and 1
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train[:, :2])
    X_test = scaler.transform(X_test[:, :2])

    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    ansatz = RealAmplitudes(feature_dimension=2, reps=1)
    backend = Aer.get_backend('aer_simulator_statevector')
    qi = QuantumInstance(backend)

    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer='COBYLA', quantum_instance=qi)
    vqc.fit(X_train, y_train)
    preds = vqc.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Quantum Model Accuracy: {acc:.2f}")
    return acc
