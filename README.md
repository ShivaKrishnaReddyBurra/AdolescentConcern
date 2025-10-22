# Predicting Adolescent Concern Over Unhealthy Food Ads

This project implements a machine learning solution to predict adolescent concern levels over unhealthy food advertisements using a Stacking Ensemble model with explainable AI (XAI) techniques. It includes a Flask-based web dashboard for interactive analysis and predictions.


## Project Overview

This repository contains a synthetic dataset and a trained model to predict adolescent concern levels (High, Medium, Low) based on demographic and ad-related features. The dashboard allows users to filter data, visualize results, and predict outcomes with feature contribution insights.

### Key Features
- **Model**: Stacking Ensemble with SMOTE for class imbalance handling.
- **Explainability**: SHAP for feature importance analysis.
- **Dashboard**: Interactive Flask app with filters, plots, and predictions.
- **Dataset**: Synthetic data with 1030 samples.

## Prerequisites

- **Python 3.8+**
- **Required Libraries**:
  - `pandas`
  - `scikit-learn`
  - `imbalanced-learn`
  - `xgboost`
  - `shap`
  - `joblib`
  - `flask`
  - `matplotlib`
  - `seaborn`

Install dependencies via pip:
```bash
pip install pandas scikit-learn imbalanced-learn xgboost shap joblib flask matplotlib
```

## Installation

1. **Clone the Repository**:
   ```bash:disable-run
   git clone https://github.com/ShivaKrishnaReddyBurra/AdolescentConcern.git
   cd AdolescentConcern
   ```

2. **Set Up Project Structure**:
   ```
   predicting-adolescent-concern/
   ├── model.py
   ├── app.py
   ├── dummy_adolescent_concern_dataset.csv
   ├── static/
   │   └── styles.css
   ├── templates/
   │   ├── dashboard.html
   │   └── predict.html
   └── README.md
   ```
## Usage

### 1. Train the Model
- Run the training script to build and save the model:
  ```bash
  python model.py
  ```
- **Output**: Displays accuracy, F1 score, global feature importances, SHAP explanations, and saves the model as `model.pkl`.

### 2. Run the Flask Application
- Start the web server:
  ```bash
  python app.py
  ```
- Open your browser and navigate to `http://127.0.0.1:5000/`.

### 3. Interact with the Dashboard
- **Filters**: Adjust age range and gender to filter the dataset.
- **Visualizations**: View concern level distribution and global feature importances.
- **Prediction**: Input demographic and ad-related data to predict concern levels and see SHAP-based feature contributions.

## File Descriptions

- **`model.py`**: Trains the Stacking Ensemble model and generates SHAP explanations.
- **`app.py`**: Flask application serving the interactive dashboard.
- **`dummy_adolescent_concern_dataset.csv`**: Synthetic dataset with 1030 samples.
- **`static/styles.css`**: Custom CSS for styling the dashboard and prediction pages.
- **`templates/dashboard.html`**: HTML template for the main dashboard.
- **`templates/predict.html`**: HTML template for displaying prediction results.
