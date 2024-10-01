import pandas as pd
from pycaret.classification import predict_model, load_model

model = load_model('lr')


def load_data(filepath):
    """
    Loading churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df, threshold=0.75):
    """
    Using the pycaret best model to make predictions on data in the df dataframe.
    Rounds up to 1 if greater than or equal to the threshold.
    """
    predictions = predict_model(model, data=df)
    predictions['Churn_prediction'] = (predictions['prediction_score'] >= threshold)
    predictions['Churn_prediction'].replace({True: '0', False: '1'}, inplace=True)
    drop_cols = predictions.columns.tolist()
    drop_cols.remove('Churn_prediction')
    return predictions.drop(drop_cols, axis=1)


def calculate_accuracy(predicted_labels, true_labels):
    """
    Calculate the accuracy of predictions.
    """
    correct_predictions = [0 if pred == true else 1 for pred, true in zip(predicted_labels, true_labels)]
    accuracy = sum(correct_predictions) / len(correct_predictions)
    return accuracy


if __name__ == "__main__":
    df = load_data(r'new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)

    # True values for comparison
    true_values = [1, 0, 0, 1, 0]
    
    # Calculating accuracy
    predicted_labels = predictions['Churn_prediction'].tolist()
    accuracy = calculate_accuracy(predicted_labels, true_values)
    
    print(f"Prediction Accuracy: {accuracy:.2f}")