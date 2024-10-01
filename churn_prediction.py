{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "49733479-1e2c-4c70-90e2-b078a6de9edf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Transformation Pipeline and Model Successfully Loaded\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "If estimator is not a Pipeline, you must run setup() first.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 30\u001b[0m\n\u001b[0;32m     28\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;18m__name__\u001b[39m \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m__main__\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m     29\u001b[0m     df \u001b[38;5;241m=\u001b[39m load_data(\u001b[38;5;124mr\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mnew_churn_data.csv\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m---> 30\u001b[0m     predictions \u001b[38;5;241m=\u001b[39m \u001b[43mmake_predictions\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdf\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     31\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpredictions:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m     32\u001b[0m     \u001b[38;5;28mprint\u001b[39m(predictions)\n",
      "Cell \u001b[1;32mIn[1], line 20\u001b[0m, in \u001b[0;36mmake_predictions\u001b[1;34m(df, threshold)\u001b[0m\n\u001b[0;32m     15\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mmake_predictions\u001b[39m(df, threshold\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.75\u001b[39m):\n\u001b[0;32m     16\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m     17\u001b[0m \u001b[38;5;124;03m    Uses the pycaret best model to make predictions on data in the df dataframe.\u001b[39;00m\n\u001b[0;32m     18\u001b[0m \u001b[38;5;124;03m    Rounds up to 1 if greater than or equal to the threshold.\u001b[39;00m\n\u001b[0;32m     19\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m---> 20\u001b[0m     predictions \u001b[38;5;241m=\u001b[39m \u001b[43mpredict_model\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodel\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdata\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdf\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     21\u001b[0m     predictions[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mChurn_prediction\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m (predictions[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mprediction_score\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m threshold)\n\u001b[0;32m     22\u001b[0m     predictions[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mChurn_prediction\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m.\u001b[39mreplace({\u001b[38;5;28;01mTrue\u001b[39;00m: \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m1\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;28;01mFalse\u001b[39;00m: \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m0\u001b[39m\u001b[38;5;124m'\u001b[39m}, inplace\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\pycaret_env1\\lib\\site-packages\\pycaret\\classification\\functional.py:2172\u001b[0m, in \u001b[0;36mpredict_model\u001b[1;34m(estimator, data, probability_threshold, encoded_labels, raw_score, round, verbose)\u001b[0m\n\u001b[0;32m   2169\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m experiment \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m   2170\u001b[0m     experiment \u001b[38;5;241m=\u001b[39m _EXPERIMENT_CLASS()\n\u001b[1;32m-> 2172\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mexperiment\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mpredict_model\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   2173\u001b[0m \u001b[43m    \u001b[49m\u001b[43mestimator\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mestimator\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2174\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdata\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdata\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2175\u001b[0m \u001b[43m    \u001b[49m\u001b[43mprobability_threshold\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mprobability_threshold\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2176\u001b[0m \u001b[43m    \u001b[49m\u001b[43mencoded_labels\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mencoded_labels\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2177\u001b[0m \u001b[43m    \u001b[49m\u001b[43mraw_score\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mraw_score\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2178\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mround\u001b[39;49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mround\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2179\u001b[0m \u001b[43m    \u001b[49m\u001b[43mverbose\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mverbose\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2180\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\pycaret_env1\\lib\\site-packages\\pycaret\\classification\\oop.py:2824\u001b[0m, in \u001b[0;36mClassificationExperiment.predict_model\u001b[1;34m(self, estimator, data, probability_threshold, encoded_labels, raw_score, round, verbose)\u001b[0m\n\u001b[0;32m   2752\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mpredict_model\u001b[39m(\n\u001b[0;32m   2753\u001b[0m     \u001b[38;5;28mself\u001b[39m,\n\u001b[0;32m   2754\u001b[0m     estimator,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   2760\u001b[0m     verbose: \u001b[38;5;28mbool\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m,\n\u001b[0;32m   2761\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m pd\u001b[38;5;241m.\u001b[39mDataFrame:\n\u001b[0;32m   2762\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   2763\u001b[0m \u001b[38;5;124;03m    This function predicts ``Label`` and ``Score`` (probability of predicted\u001b[39;00m\n\u001b[0;32m   2764\u001b[0m \u001b[38;5;124;03m    class) using a trained model. When ``data`` is None, it predicts label and\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   2821\u001b[0m \n\u001b[0;32m   2822\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m-> 2824\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43msuper\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mpredict_model\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   2825\u001b[0m \u001b[43m        \u001b[49m\u001b[43mestimator\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mestimator\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2826\u001b[0m \u001b[43m        \u001b[49m\u001b[43mdata\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdata\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2827\u001b[0m \u001b[43m        \u001b[49m\u001b[43mprobability_threshold\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mprobability_threshold\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2828\u001b[0m \u001b[43m        \u001b[49m\u001b[43mencoded_labels\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mencoded_labels\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2829\u001b[0m \u001b[43m        \u001b[49m\u001b[43mraw_score\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mraw_score\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2830\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mround\u001b[39;49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mround\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2831\u001b[0m \u001b[43m        \u001b[49m\u001b[43mverbose\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mverbose\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2832\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\pycaret_env1\\lib\\site-packages\\pycaret\\internal\\pycaret_experiment\\supervised_experiment.py:4936\u001b[0m, in \u001b[0;36m_SupervisedExperiment.predict_model\u001b[1;34m(self, estimator, data, probability_threshold, encoded_labels, raw_score, round, verbose, ml_usecase, preprocess)\u001b[0m\n\u001b[0;32m   4934\u001b[0m     pipeline\u001b[38;5;241m.\u001b[39msteps \u001b[38;5;241m=\u001b[39m pipeline\u001b[38;5;241m.\u001b[39msteps[:\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]\n\u001b[0;32m   4935\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_setup_ran:\n\u001b[1;32m-> 4936\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m   4937\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mIf estimator is not a Pipeline, you must run setup() first.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   4938\u001b[0m     )\n\u001b[0;32m   4939\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   4940\u001b[0m     pipeline \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpipeline\n",
      "\u001b[1;31mValueError\u001b[0m: If estimator is not a Pipeline, you must run setup() first."
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from pycaret.classification import predict_model, load_model\n",
    "\n",
    "model = load_model('lr')\n",
    "\n",
    "\n",
    "def load_data(filepath):\n",
    "    \"\"\"\n",
    "    Loads churn data into a DataFrame from a string filepath.\n",
    "    \"\"\"\n",
    "    df = pd.read_csv(filepath, index_col='customerID')\n",
    "    return df\n",
    "\n",
    "\n",
    "def make_predictions(df, threshold=0.75):\n",
    "    \"\"\"\n",
    "    Uses the pycaret best model to make predictions on data in the df dataframe.\n",
    "    Rounds up to 1 if greater than or equal to the threshold.\n",
    "    \"\"\"\n",
    "    predictions = predict_model(model, data=df)\n",
    "    predictions['Churn_prediction'] = (predictions['prediction_score'] >= threshold)\n",
    "    predictions['Churn_prediction'].replace({True: '1', False: '0'}, inplace=True)\n",
    "    drop_cols = predictions.columns.tolist()\n",
    "    drop_cols.remove('Churn_prediction')\n",
    "    return predictions.drop(drop_cols, axis=1)\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    df = load_data(r'new_churn_data.csv')\n",
    "    predictions = make_predictions(df)\n",
    "    print('predictions:')\n",
    "    print(predictions)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
