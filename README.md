Welcome to the ChurnPredictor project! This repository contains the code and resources needed to develop a machine learning model for customer churn prediction using PyCaret.

Project Overview

This project aims to accurately predict customer churn utilizing PyCaret, an efficient Python library for machine learning. The primary activities include:

Preparing and exploring the dataset for insights.
Training several machine learning models to identify the best performer.
Saving the optimized model for future predictions.
Implementing a Python script to predict churn probabilities on new datasets.
Directory Structure

notebooks/: Includes the Jupyter Notebook used for training and evaluating the model.
scripts/: Contains the Python script (churn_predictor.py) for predicting churn probabilities on new datasets.
data/: Holds the datasets, specifically your_data.csv for model training and new_churn_data.csv for testing.
README.md: This file, providing an overview of the project.
Installation

To set up the environment and run this project, ensure you have Python installed and then execute:

pip install pycaret pandas

Usage

Model Training

Open the Jupyter Notebook located in the notebooks/ directory.
Execute the cells to load data, train models, and select the best-performing model using PyCaret.
The best model will be automatically saved as best_churn_model.pkl.
Making Predictions

Use the Python script churn_predictor.py in the scripts/ directory to predict churn probabilities on new data:

Ensure new_churn_data.csv is present in the data/ directory.
Run the script:
python scripts/churn_predictor.py

This will output the churn probabilities for each entry in the test dataset.

Evaluation

Compare predicted probabilities from the script against the known true values [1, 0, 0, 1, 0] to assess model accuracy.

Conclusion

This project demonstrates the use of automated machine learning tools provided by PyCaret to streamline the process of developing a robust churn prediction model.

Contributing

Contributions are welcome! Please fork this repository and submit a pull request if you have any improvements or suggestions.

License

This project is licensed under the MIT License. For more details, see the LICENSE file. 
