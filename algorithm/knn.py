import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Charger le dataset
    data = pd.read_csv("undersampled_data.csv")
    X = data[['Time', 'V1', 'V2', 'Amount']]
    y = data['Class']

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser et ajuster le modèle KNN
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prédictions
    y_pred = knn.predict(X_test)

    # Afficher les résultats réels et prédits
    print("Résultats réels vs prédits :")
    for pred, actual in zip(y_pred, y_test):
        print(f"Prédit : {pred}, Réel : {actual}")

    # Évaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
