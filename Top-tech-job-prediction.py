import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
from sklearn.metrics import f1_score, recall_score, precision_score

# Load and preprocess the data
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.columns)
    if 'count' not in data.columns:
        data['count'] = 1
    
    unique_positions_counts = data['position'].value_counts().reset_index()
    unique_positions_counts.columns = ['position', 'count']
    
    return unique_positions_counts

# Target encoding for categorical data
def encode_positions(data):
    original_positions = data['position']
    encoder = TargetEncoder(cols=['position'])
    encoded_data = encoder.fit_transform(data, data['count'])
    return encoded_data, original_positions

# Split data into training and testing sets
def split_data(encoded_data):
    X = encoded_data.drop('count', axis=1)
    y = (encoded_data['count'] > 1).astype(int)  # Binary classification example
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the RandomForestClassifier
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    return predictions

# Main function
def main():
    file_path = "cleaned_data.csv"
    data = load_and_preprocess(file_path)
    
    encoded_data, original_positions = encode_positions(data)
    
    X_train, X_test, y_train, y_test = split_data(encoded_data)
    
    model = train_random_forest(X_train, y_train)
    
    predictions = evaluate_model(model, X_test, y_test)
    print("Model evaluation complete.")

# Run the script
if __name__ == "__main__":
    main()
