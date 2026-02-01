# Fertilizer Recommendation System

A machine learning project designed to recommend the optimal fertilizer for crops based on soil composition and environmental factors.

## Project Structure

This project is organized as follows:

- **`notebooks/`**: Jupyter notebooks for data exploration, analysis, and model training.
  - `fertilizer_recommendation_system.ipynb`: Main notebook containing the entire workflow from EDA to model training.
- **`data/`**: Dataset files.
  - `fertilizer_prediction.csv`: The source dataset containing soil and crop data.
- **`models/`**: Serialized machine learning models and encoders.
  - `fertilizer_voting_model.pkl`: The trained Voting Classifier model used for prediction.
  - `fertilizer_rf_model.pkl`: Random Forest model.
  - `fertilizer_scaler.pkl`: Scaler for numerical features.
  - `*_encoder.pkl`: Label encoders for categorical variables.
- **`app/`**: Streamlit web application.
  - `streamlit_app.py`: The user interface for interacting with the model.
- **`docs/`**: Documentation and generated reports/graphs.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn streamlit
    ```

## Usage

### Running the Web App
To explore the model predictions through the interactive dashboard, run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

### Training the Model
To reproduce the analysis or retrain the model, open the notebook in `notebooks/`:

```bash
jupyter notebook notebooks/fertilizer_recommendation_system.ipynb
```

## Features
- **Data Analysis**: Comprehensive EDA of soil and crop attributes.
- **Model Training**: Evaluation of multiple algorithms including Random Forest and Voting Classifiers.
- **Interactive UI**: User-friendly interface for real-time recommendations.

## License
MIT
