import numpy as np
import pandas as pd
from collections import Counter


# Function to split data into training and test sets
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[indices[:split_idx]], X.iloc[indices[split_idx:]]
    y_train, y_test = y.iloc[indices[:split_idx]], y.iloc[indices[split_idx:]]
    return X_train, X_test, y_train, y_test


# Function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Class for implementing a k-NN classifier
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    # Function to store training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Function to predict classes for a set of test samples
    def predict(self, X):
        y_pred = [self._predict(x) for x in X.to_numpy()]
        return np.array(y_pred)

    # Helper function to predict the class for a single test sample
    def _predict(self, x):
        # Calculate distances from the test sample to all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train.to_numpy()]
        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        # Determine the most common class label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Function to calculate the accuracy of predictions
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Function to generate a classification report
def classification_report(y_true, y_pred):
    labels = np.unique(y_true)
    report = {}
    for label in labels:
        tp = sum((y_pred == label) & (y_true == label))  # True positives
        fp = sum((y_pred == label) & (y_true != label))  # False positives
        fn = sum((y_pred != label) & (y_true == label))  # False negatives
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        report[label] = {"precision": precision, "recall": recall, "f1-score": f1_score}
    return report


# Function to generate a confusion matrix
def confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, label_true in enumerate(labels):
        for j, label_pred in enumerate(labels):
            matrix[i, j] = sum((y_true == label_true) & (y_pred == label_pred))
    return matrix


def main():
    # Load the dataset
    data = pd.read_csv("undersampled_data.csv")
    X = data[['Time', 'V1', 'V2', 'Amount']]
    y = data['Class']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the KNN model
    k = 3
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Display actual and predicted results
    print("Actual vs Predicted Results:")
    for pred, actual in zip(y_pred, y_test):
        print(f"Predicted: {pred}, Actual: {actual}")
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test.to_numpy(), y_pred))
    print("Classification Report:\n", classification_report(y_test.to_numpy(), y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test.to_numpy(), y_pred))


if __name__ == "__main__":
    main()
