# Customer Churn Prediction

Developed a machine learning model to predict customer churn in a telecommunications company. The project leverages advanced classification algorithms to identify customers likely to discontinue services, enabling proactive retention strategies and improved customer satisfaction.

## Tech Stack
- **Python**: Core programming language
- **scikit-learn**: Machine learning model development and evaluation
- **PyCaret**: Automated machine learning workflows
- **Pandas**: Data manipulation and analysis

## Key Features
- Automated ML pipeline for model selection and optimization
- Real-time churn prediction capabilities
- Comprehensive data preprocessing and feature engineering
- Model evaluation with accuracy metrics
- Easy-to-use prediction interface

## Project Structure
```
├── churn_prediction.py     # Main prediction script
├── predict_churn_data.py   # Prediction utilities
├── requirements.txt        # Project dependencies
├── new_churn_data.csv     # Sample data for predictions
└── lr.pkl                 # Trained logistic regression model
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChurnModeling.git
cd ChurnModeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run predictions:
```bash
python predict_churn_data.py
```

## Model Details
The project uses a Logistic Regression model trained on historical customer data. Features include:
- Customer demographics
- Service usage patterns
- Billing information
- Contract details

## Usage
To make predictions on new customer data:
1. Prepare your data in CSV format with required features
2. Use the prediction script:
```python
from predict_churn_data import make_predictions
predictions = make_predictions(your_data)
```

## Performance
- Model achieves high accuracy in identifying potential churners
- Fast prediction times for real-time applications
- Robust handling of various input data formats

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
