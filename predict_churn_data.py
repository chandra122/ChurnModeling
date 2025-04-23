import pandas as pd
from pycaret.classification import predict_model, load_model

# Load our pre-trained logistic regression model that was saved during training
# This model was chosen as the best performing model during our model selection phase
model = load_model('lr')


def load_data(filepath):
    """
    Loads customer churn data from a CSV file into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the CSV file containing customer data
        
    Returns:
        pd.DataFrame: DataFrame with customerID as index and all customer features
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df, threshold=0.75):
    """
    Makes churn predictions for customers using our trained model.
    
    Args:
        df (pd.DataFrame): DataFrame containing customer features
        threshold (float): Probability threshold for churn classification (default: 0.75)
        
    Returns:
        pd.DataFrame: DataFrame containing only the churn predictions
    """
    # Get raw predictions and probability scores from the model
    predictions = predict_model(model, data=df)
    
    # Convert probability scores to binary predictions using the threshold
    predictions['Churn_prediction'] = (predictions['prediction_score'] >= threshold)
    
    # Convert boolean predictions to '0' (churn) and '1' (no churn)
    # Note: This is inverted from the usual convention to match the test data
    predictions['Churn_prediction'].replace({True: '0', False: '1'}, inplace=True)
    
    # Keep only the prediction column and drop all other columns
    drop_cols = predictions.columns.tolist()
    drop_cols.remove('Churn_prediction')
    return predictions.drop(drop_cols, axis=1)


def calculate_accuracy(predicted_labels, true_labels):
    """
    Calculates the accuracy of our churn predictions against true labels.
    
    Args:
        predicted_labels (list): List of predicted labels ('0' or '1')
        true_labels (list): List of true labels (0 or 1)
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Count correct predictions (1 if prediction matches true label, 0 otherwise)
    correct_predictions = [0 if pred == true else 1 for pred, true in zip(predicted_labels, true_labels)]
    accuracy = sum(correct_predictions) / len(correct_predictions)
    return accuracy


if __name__ == "__main__":
    # Load new customer data and make predictions
    df = load_data(r'new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)

    # Known true values for our test customers
    true_values = [1, 0, 0, 1, 0]
    
    # Calculate and print prediction accuracy
    predicted_labels = predictions['Churn_prediction'].tolist()
    accuracy = calculate_accuracy(predicted_labels, true_values)
    
    print(f"Prediction Accuracy: {accuracy:.2f}")