import pandas as pd  # For data manipulation
from imblearn.over_sampling import SMOTE  # For oversampling
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets


def main():
    # Loading dataset
    df = pd.read_csv("creditcard.csv")  # Load your dataset

    # Splitting attributes
    x = df.drop("Class", axis=1)  # All attributes except "Class"
    y = df["Class"]

    # Split data into 80% training and 20% testing, with seed for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)

    # Check data distribution before resampling
    print(f"Before resampling:\n{y_train.value_counts()}")

    # Apply SMOTE for oversampling the minority class
    smote = SMOTE(sampling_strategy="minority", random_state=69)
    xResample, yResample = smote.fit_resample(x_train, y_train)
    print(f"After oversample with SMOTE:\n{yResample.value_counts()}")

    # Save undersampled data to CSV
    oversampled_data = pd.DataFrame(xResample, columns=x.columns)
    oversampled_data['Class'] = yResample  # Adding the target column
    oversampled_data.to_csv("oversampled_data.csv", index=False)
    print("oversampled data has been saved to 'oversampled_data.csv'")


if __name__ == "__main__":
    main()
