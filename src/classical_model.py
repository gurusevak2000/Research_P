from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_classical_model():
    # Load the iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Binary classification (use only 2 classes)
    X, y = X[y != 2], y[y != 2]  # Remove class 2 (we’re only using classes 0 and 1)
    
    # Split the data into training and test sets (70% training, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create an SVM classifier with the Radial Basis Function (RBF) kernel
    model = SVC(kernel='rbf')
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy by comparing predicted values with true values
    acc = accuracy_score(y_test, y_pred)
    print(f"Classical Model Accuracy: {acc:.2f}")
    return acc
