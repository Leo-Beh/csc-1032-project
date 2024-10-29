# S3: Algorithm Description

- **Objective**:
    - To implement a k-Nearest Neighbors (k-NN) algorithm for classification tasks without using specialized libraries
      like `sklearn`.

- **Dataset**:
    - The dataset is loaded from a CSV file named `undersampled_data.csv`.
    - Features used for prediction include `Time`, `V1`, `V2`, and `Amount`.
    - The target variable is `Class`.

- **Data Preprocessing**:
    - A custom function `train_test_split` is defined to randomly shuffle and split the dataset into training and
      testing sets:
      ```python
      def train_test_split(X, y, test_size=0.2, random_state=42):
          np.random.seed(random_state)
          indices = np.arange(len(X))
          np.random.shuffle(indices)
  
          split_idx = int(len(X) * (1 - test_size))
          X_train, X_test = X.iloc[indices[:split_idx]], X.iloc[indices[split_idx:]]
          y_train, y_test = y.iloc[indices[:split_idx]], y.iloc[indices[split_idx:]]
          
          return X_train, X_test, y_train, y_test
      ```

- **Distance Calculation**:
    - The algorithm uses the Euclidean distance metric to determine the proximity of data points:
      ```python
      def euclidean_distance(x1, x2):
          return np.sqrt(np.sum((x1 - x2) ** 2))
      ```

- **Model Implementation**:
    - A `KNNClassifier` class is defined to encapsulate the k-NN algorithm:
      ```python
      class KNNClassifier:
          def __init__(self, k=3):
              self.k = k
  
          def fit(self, X, y):
              self.X_train = X
              self.y_train = y
  
          def predict(self, X):
              y_pred = [self._predict(x) for x in X.to_numpy()]
              return np.array(y_pred)
  
          def _predict(self, x):
              distances = [euclidean_distance(x, x_train) for x_train in self.X_train.to_numpy()]
              k_indices = np.argsort(distances)[:self.k]
              k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
              most_common = Counter(k_nearest_labels).most_common(1)
              return most_common[0][0]
      ```

- **Evaluation Metrics**:
    - Functions are defined to calculate:
        - **Accuracy**: The proportion of correct predictions:
          ```python
          def accuracy_score(y_true, y_pred):
              return np.mean(y_true == y_pred)
          ```
        - **Classification Report**: Includes precision, recall, and F1-score for each class:
          ```python
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
          ```
        - **Confusion Matrix**: A table that describes the performance of the classification model:
          ```python
          def confusion_matrix(y_true, y_pred):
              labels = np.unique(y_true)
              matrix = np.zeros((len(labels), len(labels)), dtype=int)
              for i, label_true in enumerate(labels):
                  for j, label_pred in enumerate(labels):
                      matrix[i, j] = sum((y_true == label_true) & (y_pred == label_pred))
              return matrix
          ```

# S4: Results and Your Analysis

- **Model Performance**:
    - The model's accuracy is calculated and printed, providing a measure of how well the k-NN algorithm performs on the
      test dataset.

- **Predictions**:
    - Actual vs predicted results are printed for manual verification and comparison:
      ```python
      for pred, actual in zip(y_pred, y_test):
          print(f"Predicted: {pred}, Actual: {actual}")
      ```

- **Classification Report**:
    - The report displays precision, recall, and F1-score for each class, allowing for a deeper understanding of model
      performance beyond mere accuracy.

- **Confusion Matrix**:
    - The confusion matrix provides insights into the types of errors made by the model (e.g., false positives and false
      negatives).

- **Insights and Observations**:
    - Analyze trends in the data, such as which classes are more often misclassified.
    - Discuss potential reasons for model performance (e.g., feature relevance, data quality).
    - Consider the implications of the results for the application of the k-NN algorithm in real-world scenarios, such
      as its sensitivity to outliers and the choice of `k`.
