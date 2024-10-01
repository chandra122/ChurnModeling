import pandas as pd
from pycaret.classification import predict_model, load_model

# Function to load data from a CSV file
def load_data(filepath):
    """
    Loads churn data into a DataFrame from a specified filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df

# Function to make predictions using a pre-trained model
def make_predictions(df, threshold=0.75):
    """
    Uses a loaded PyCaret model to make predictions on the dataframe.
    Rounds up to 1 if prediction confidence is greater than or equal to the threshold.
    """
    # Load your pre-trained model
    model = load_model('lr')  # Ensure the 'lr' model file is in an accessible location
    
    # Make predictions
    predictions = predict_model(model, data=df)

    # Set the prediction threshold and adjust predictions accordingly
    predictions['Churn_prediction'] = (predictions['prediction_score'] >= threshold).astype(int)
    predictions = predictions.rename(columns={'prediction_score': 'Score'})

    # Keep only the columns necessary for output
    return predictions[['Score', 'Churn_prediction']]

if __name__ == "__main__":
    # Load new data for prediction
    df_new = load_data('new_churn_data.csv')
    
    # Print the head of the data to confirm correct load
    print("Loaded New Data:")
    print(df_new.head())

    # Get and print predictions
    predictions = make_predictions(df_new)
    print('Predictions:')
    print(predictions)

    # True values for comparison (from the problem statement)
    true_values = [1, 0, 0, 1, 0]
    
    # Calculate accuracy
    predicted_labels = predictions['Churn_prediction'].tolist()
    correct_predictions = [1 if pred == true else 0 for pred, true in zip(predicted_labels, true_values)]
    accuracy = sum(correct_predictions) / len(correct_predictions)
    print(f"Prediction Accuracy: {accuracy:.2f}")
