## S3: Algorithm Description

- **Objective**:
    - To implement a k-Nearest Neighbors (k-NN) algorithm for classification tasks without using specialized libraries
      like `sklearn`.

- **Dataset**:
    - The dataset is loaded from a CSV file named `undersampled_data.csv`.
    - Features used for prediction include `Time`, `V1`, `V2`, and `Amount`.
    - The target variable is `Class`.

- **Data Preprocessing**:
    - A custom function `train_test_split` is defined to randomly shuffle and split the dataset into training and
      testing sets.
    - The split ratio is set to 80% for training and 20% for testing.

- **Distance Calculation**:
    - The algorithm uses the Euclidean distance metric to determine the proximity of data points.
    - A helper function `euclidean_distance` calculates the distance between two points.

- **Model Implementation**:
    - A `KNNClassifier` class is defined to encapsulate the k-NN algorithm.
    - The class includes:
        - A constructor to initialize the number of neighbors (`k`).
        - A `fit` method to store the training data.
        - A `predict` method that generates predictions for the test data.
        - A private `_predict` method that computes the class for a single sample based on the nearest neighbors'
          labels.

- **Evaluation Metrics**:
    - Functions are defined to calculate:
        - **Accuracy**: The proportion of correct predictions.
        - **Classification Report**: Includes precision, recall, and F1-score for each class.
        - **Confusion Matrix**: A table that describes the performance of the classification model.

## S4: Results and Your Analysis

- **Model Performance**:
    - The model's accuracy is calculated and printed, providing a measure of how well the k-NN algorithm performs on the
      test dataset.

- **Predictions**:
    - Actual vs predicted results are printed for manual verification and comparison.
    - This helps to visualize where the model performs well and where it misclassifies.

- **Classification Report**:
    - The report displays precision, recall, and F1-score for each class, allowing for a deeper understanding of model
      performance beyond mere accuracy.
    - This helps in identifying any class imbalances or performance issues for specific classes.

- **Confusion Matrix**:
    - The confusion matrix provides insights into the types of errors made by the model (e.g., false positives and false
      negatives).
    - It allows for the identification of patterns in misclassification.

- **Insights and Observations**:
    - Analyze trends in the data, such as which classes are more often misclassified.
    - Discuss potential reasons for model performance (e.g., feature relevance, data quality).
    - Consider the implications of the results for the application of the k-NN algorithm in real-world scenarios, such
      as its sensitivity to outliers and the choice of `k`.
