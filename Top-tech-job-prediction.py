import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder

# Set up logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Dataset loaded from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}. Please check the file path.")
        raise
    except pd.errors.ParserError:
        logging.error(f"Error parsing the file at {file_path}. Please check its format.")
        raise

def preprocess_data(data, categorical_columns, target_column):
    """
    Preprocess the dataset by encoding categorical variables and ensuring the target column exists.
    """
    try:
        # Create the 'count' column if missing
        if target_column not in data.columns:
            logging.info(f"'{target_column}' column not found. Creating it as job counts grouped by 'position'.")
            data = data.groupby('position').size().reset_index(name=target_column)

        # Target encode or one-hot encode categorical variables
        encoded_data = data.copy()

        for col in categorical_columns:
            if col in encoded_data.columns and encoded_data[col].dtype == 'object':
                # One-hot encode low-cardinality features
                if encoded_data[col].nunique() <= 10:
                    encoded_data = pd.get_dummies(encoded_data, columns=[col], drop_first=True)
                    logging.info(f"One-hot encoded column: {col}.")
                else:
                    # Target encode high-cardinality features
                    encoder = TargetEncoder(cols=[col])
                    encoded_data = encoder.fit_transform(encoded_data, encoded_data[target_column])
                    logging.info(f"Target encoded column: {col}.")

        logging.info("Finished encoding categorical variables.")
        return encoded_data
    except KeyError as e:
        logging.error(f"KeyError in preprocessing data: {e}. Ensure column names are correct.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in preprocessing data: {e}.")
        raise

def split_data(data, target_column):
    """
    Split the dataset into training and testing sets.
    """
    try:
        X = data.drop(columns=[target_column])  # Features
        y = data[target_column]  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logging.info("Split data into training and testing sets.")
        return X_train, X_test, y_train, y_test
    except KeyError as e:
        logging.error(f"KeyError in splitting data: {e}. Ensure the target column exists.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in splitting data: {e}.")
        raise

def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest model for regression.
    """
    try:
        rf_model = RandomForestRegressor(random_state=42, **kwargs)
        rf_model.fit(X_train, y_train)
        logging.info("Trained Random Forest model.")
        return rf_model
    except Exception as e:
        logging.error(f"Error in training Random Forest model: {e}.")
        raise

def evaluate_model(rf_model, X_test, y_test):
    """
    Evaluate the trained Random Forest model.
    """
    try:
        # Make predictions
        predictions = rf_model.predict(X_test)

        # Compute Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Mean Squared Error (MSE): {mse}")

        return predictions
    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}.")
        raise

def get_top_jobs(test_data, predictions, top_n=10):
    """
    Identify the top jobs based on predicted counts.
    """
    try:
        # Add predictions to the test data
        test_data['predicted_count'] = predictions

        # Sort by predicted count and select top jobs
        top_jobs = test_data.sort_values(by='predicted_count', ascending=False).head(top_n)
        logging.info("Identified top jobs based on predicted counts.")
        return top_jobs
    except Exception as e:
        logging.error(f"Error in identifying top jobs: {e}.")
        raise


def main():
    """
    Main function to orchestrate the workflow.
    """
    try:
        # Load dataset
        file_path = 'cleaned_data.csv'  # Update this path as needed
        data = load_dataset(file_path)

        # Define categorical columns and target column
        categorical_columns = ['company_name', 'position', 'location', 'contract_type', 'language', 'job_description']
        target_column = 'count'

        # Preprocess data
        encoded_data = preprocess_data(data, categorical_columns, target_column)

        # Verify the 'count' column
        if target_column not in encoded_data.columns:
            raise KeyError(f"'{target_column}' column is still missing after preprocessing!")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(encoded_data, target_column)

        # Train Random Forest model
        rf_model = train_random_forest(X_train, y_train,
                                       n_estimators=100,
                                       max_depth=10,
                                       min_samples_split=5,
                                       min_samples_leaf=2,
                                       max_features='sqrt')

        # Evaluate the model
        predictions = evaluate_model(rf_model, X_test, y_test)

        # Get top jobs for 2025 based on X_test
        X_test['position'] = data['position'].iloc[X_test.index]  # Reattach 'position' for reference
        top_jobs = get_top_jobs(X_test, predictions, top_n=10)
        print("Top Jobs for 2025:")
        print(top_jobs)
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}.")
        raise

if __name__ == "__main__":
    main()
