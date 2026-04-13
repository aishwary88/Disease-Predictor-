import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="🏥 Disease Predictor", layout="wide")

st.title("🏥 Disease Prediction System")
st.write("Select your symptoms to get a disease prediction")

try:
    # Load model and encoder
    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    
    # Load dataset to get feature names
    df = pd.read_csv("Dataset1.csv")
    features = [col for col in df.columns if col != "diseases"]
    
    st.success(f"✅ Model loaded | {len(features)} symptoms available")
    
    # Create columns for checkboxes
    st.subheader("Select Your Symptoms:")
    cols = st.columns(3)
    selected_symptoms = []
    
    for idx, feature in enumerate(features):
        col = cols[idx % 3]
        if col.checkbox(feature):
            selected_symptoms.append(feature)
    
    # Show selected symptoms count
    st.info(f"Selected: {len(selected_symptoms)} symptom(s)")
    
    # Prediction
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Predict Disease", use_container_width=True):
            if selected_symptoms:
                # Create input array
                input_data = []
                for feature in features:
                    input_data.append(1 if feature in selected_symptoms else 0)
                
                # Make prediction
                prediction = model.predict([input_data])[0]
                probability = model.predict_proba([input_data])[0].max()
                disease_name = encoder.inverse_transform([prediction])[0]
                
                # Display results
                st.success(f"🎯 **Predicted Disease:** {disease_name}")
                st.metric("Confidence", f"{probability:.2%}")
                
            else:
                st.warning("⚠️ Please select at least one symptom")
    
    with col2:
        if st.button("🔄 Clear Selection", use_container_width=True):
            st.rerun()
    
    # Info section
    st.divider()
    with st.expander("ℹ️ About This System"):
        st.write("""
        - **Model:** Random Forest Classifier
        - **Features:** 377 symptoms
        - **Purpose:** Educational disease prediction
        - **Disclaimer:** This is not a replacement for professional medical advice
        """)

except FileNotFoundError:
    st.error("❌ Model files not found! Run `python train.py` first.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
import streamlit as st
import pickle
import pandas as pd

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Load dataset to get feature names
df = pd.read_csv("Dataset1.csv")
features = [col for col in df.columns if col != "diseases"]

st.title("🏥 Disease Prediction System")
st.write("Select symptoms to predict the disease")

# Create checkboxes for each symptom
selected_symptoms = []
cols = st.columns(3)
for idx, feature in enumerate(features):
    col = cols[idx % 3]
    if col.checkbox(feature):
        selected_symptoms.append(feature)

# Make prediction
if st.button("Predict Disease"):
    if selected_symptoms:
        # Create input array
        input_data = []
        for feature in features:
            input_data.append(1 if feature in selected_symptoms else 0)
        
        # Make prediction
        prediction = model.predict([input_data])[0]
        disease_name = encoder.inverse_transform([prediction])[0]
        
        st.success(f"🎯 Predicted Disease: **{disease_name}**")
    else:
        st.warning("Please select at least one symptom")
