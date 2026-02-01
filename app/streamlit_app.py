# ========================== STREAMLIT APP ==========================
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================== LOAD MODELS, ENCODERS, AND DATA ==========================
import os

# Get the absolute path to the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script
model_path = os.path.join(base_dir, '../models/fertilizer_voting_model.pkl')
scaler_path = os.path.join(base_dir, '../models/fertilizer_scaler.pkl')
target_encoder_path = os.path.join(base_dir, '../models/fertilizer_target_encoder.pkl')
soil_encoder_path = os.path.join(base_dir, '../models/soil_encoder.pkl')
crop_encoder_path = os.path.join(base_dir, '../models/crop_encoder.pkl')
data_path = os.path.join(base_dir, '../data/fertilizer_prediction.csv')

# ========================== LOAD MODELS, ENCODERS, AND DATA ==========================
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(target_encoder_path, 'rb') as f:
    le_target = pickle.load(f)

with open(soil_encoder_path, 'rb') as f:
    le_soil = pickle.load(f)

with open(crop_encoder_path, 'rb') as f:
    le_crop = pickle.load(f)

# Load dataset for visualization
df = pd.read_csv(data_path)

# ========================== STREAMLIT PAGE CONFIG ==========================
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ========================== HEADER ==========================
st.title("🌱 Fertilizer Recommendation System")
st.markdown("""
Predict the **most suitable fertilizer** for your crop based on soil and environmental conditions.  
Explore **fertilizer, soil, crop, and nutrient distributions** below.
""")
st.markdown("---")

# ========================== USER INPUTS ==========================
st.sidebar.header("Enter Soil & Environmental Parameters")
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0)
moisture = st.sidebar.number_input("Soil Moisture (%)", 0.0, 100.0, 30.0)
nitrogen = st.sidebar.number_input("Nitrogen (N)", 0, 100, 20)
phosphorous = st.sidebar.number_input("Phosphorous (P)", 0, 100, 20)
potassium = st.sidebar.number_input("Potassium (K)", 0, 100, 20)

soil_type = st.sidebar.selectbox("Soil Type", le_soil.classes_)
crop_type = st.sidebar.selectbox("Crop Type", le_crop.classes_)

# ========================== PREDICTION FUNCTION ==========================
def recommend_fertilizer(temp, hum, moist, n, p, k, soil, crop):
    soil_encoded = le_soil.transform([soil])[0]
    crop_encoded = le_crop.transform([crop])[0]
    
    input_array = np.array([[temp, hum, moist, n, p, k, soil_encoded, crop_encoded]])
    input_scaled = scaler.transform(input_array)
    
    pred = model.predict(input_scaled)
    top_probs = model.predict_proba(input_scaled)[0]
    
    top_idx = np.argsort(top_probs)[::-1][:3]  # Top 3 fertilizers
    top_ferts = le_target.inverse_transform(top_idx)
    top_values = top_probs[top_idx]
    
    return le_target.inverse_transform(pred)[0], top_ferts, top_values

# ========================== PREDICTION BUTTON ==========================
if st.button("Predict Fertilizer"):
    fertilizer, top3, top_probs = recommend_fertilizer(
        temperature, humidity, moisture, nitrogen, phosphorous, potassium, soil_type, crop_type
    )
    
    st.success(f"🎯 Recommended Fertilizer: **{fertilizer}**")
    
    # Display top 3 as table
    st.subheader("Top 3 Fertilizer Recommendations")
    top_df = pd.DataFrame({
        'Fertilizer': top3,
        'Probability': [f"{p*100:.2f}%" for p in top_probs]
    })
    st.table(top_df)
    
    # Bar chart for top 3 probabilities
    fig, ax = plt.subplots()
    sns.barplot(x=top_probs, y=top3, palette='viridis', ax=ax)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Fertilizer")
    ax.set_title("Top 3 Fertilizer Probabilities")
    st.pyplot(fig)

# ========================== DATA VISUALIZATIONS ==========================
st.header("📊 Fertilizer & Crop Insights")
st.markdown("Explore fertilizer, soil, and crop distributions in your dataset.")

# Tabs for interactive visualization
tab1, tab2, tab3, tab4 = st.tabs(["Fertilizer Distribution", "Soil Distribution", "Crop Distribution", "NPK Distributions"])

with tab1:
    st.subheader("🌱 Fertilizer Type Distribution")
    fert_counts = df['Fertilizer Name'].value_counts()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=fert_counts.index, y=fert_counts.values, palette='coolwarm', ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Fertilizer Name")
    ax.set_title("Fertilizer Count Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab2:
    st.subheader("🌍 Soil Type Distribution")
    soil_counts = df['Soil Type'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(soil_counts.values, labels=soil_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    ax.set_title("Soil Type Distribution")
    st.pyplot(fig)

with tab3:
    st.subheader("🌾 Crop Type Distribution")
    crop_counts = df['Crop Type'].value_counts()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=crop_counts.index, y=crop_counts.values, palette='Greens', ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Crop Type")
    ax.set_title("Crop Type Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab4:
    st.subheader("🧪 NPK Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    sns.histplot(df['Nitrogen'], bins=20, color='lightcoral', ax=axes[0])
    axes[0].set_title("Nitrogen Distribution")
    sns.histplot(df['Phosphorous'], bins=20, color='skyblue', ax=axes[1])
    axes[1].set_title("Phosphorous Distribution")
    sns.histplot(df['Potassium'], bins=20, color='lightgreen', ax=axes[2])
    axes[2].set_title("Potassium Distribution")
    st.pyplot(fig)

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("""
Made with ❤️ by **Luvish, Sanchit and Arsh**  
Using Python, Scikit-Learn, XGBoost, and Streamlit
""")
