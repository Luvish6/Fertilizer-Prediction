# Fertilizer Prediction System 

A machine learning–based decision support system that recommends the most suitable fertilizer for crops by analyzing soil composition, crop type, and environmental factors. The project focuses on improving agricultural productivity through data-driven fertilizer selection.

---
## Overview

Efficient fertilizer usage is essential for sustainable agriculture. Manual fertilizer selection often leads to overuse or incorrect application, negatively impacting soil health and crop yield. This project applies supervised machine learning techniques to predict the optimal fertilizer, enabling informed and precise agricultural decisions.
---

## Objectives

- Analyze soil and crop-related parameters influencing fertilizer selection  
- Build and evaluate machine learning models for fertilizer prediction  
- Improve prediction accuracy using ensemble learning techniques  
- Provide a simple and interactive interface for real-time recommendations  

---

## Methodology

- Data preprocessing including encoding and feature scaling  
- Exploratory data analysis to identify trends and correlations  
- Training and comparison of multiple classification models  
- Implementation of an ensemble-based Voting Classifier  
- Deployment using a Streamlit web application  

---

## Project Structure
Fertilizer-Prediction/
│
├── notebooks/
│ └── fertilizer_prediction.ipynb
│
├── data/
│ └── fertilizer_prediction.csv
│
├── models/
│ ├── fertilizer_model.pkl
│ └── encoders.pkl
│
├── app/
│ └── streamlit_app.py
│
├── docs/
│ └── results_and_visualizations/
│
└── README.md

---

## Technologies Used

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn  
- **Machine Learning:** Random Forest, Ensemble Learning (Voting Classifier)  
- **Web Framework:** Streamlit  
- **Version Control:** Git and GitHub  

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Luvish6/Fertilizer-Prediction.git
cd Fertilizer-Prediction
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
streamlit run app/streamlit_app.py



