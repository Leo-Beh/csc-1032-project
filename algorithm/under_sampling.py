import pandas as pd  # For data manipulation
from imblearn.over_sampling import SMOTE  # For oversampling
from imblearn.under_sampling import RandomUnderSampler  # For undersampling
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

    # Apply Random UnderSampling to balance by undersampling the majority class
    undersample = RandomUnderSampler(sampling_strategy="majority", random_state=69)

    x_resample, y_resample = undersample.fit_resample(x_train, y_train)
    print(f"After undersample with RandomUnderSampler:\n{y_resample.value_counts()}")

    # Save undersampled data to CSV
    undersampled_data = pd.DataFrame(x_resample, columns=x.columns)
    undersampled_data['Class'] = y_resample  # Adding the target column
    undersampled_data.to_csv("undersampled_data.csv", index=False)
    print("Undersampled data has been saved to 'undersampled_data.csv'")


if __name__ == "__main__":
    main()
