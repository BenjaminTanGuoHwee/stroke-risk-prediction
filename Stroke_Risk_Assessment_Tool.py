import streamlit as st
import pandas as pd
import joblib

# Load model and expected columns
model = joblib.load("stroke_prediction_model.joblib")
model_columns = joblib.load("model_columns.joblib")  # Must be saved during training!

# UI
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.image("medical_icon.png", width=130)  # Add relevant medical icon
with col2:
    st.title("Stroke Risk Assessment Tool🔧")
    
st.markdown("Designed for Healthcare Professionals & Individuals")
st.caption("Powered by the HADAPS Model v1.3.6")

# Add divider
st.divider()

# Disclaimers
with st.expander("⚠️ **Important Disclaimer**", expanded=True):
    st.warning("""
    - This tool provides **preliminary risk assessment only** and is not a substitute for professional medical evaluation.  
    - The underlying model uses statistical and machine learning techniques to estimate stroke risk based on input data. **No prediction is 100% accurate**, false positives and false negatives may occur. 
    - Always consult a qualified healthcare provider for any medical concerns.""")

st.divider()

# Input form
with st.form("risk_form"):
    st.header("A. Patient Demographics")
    age_cat = st.selectbox("Age Category", ["--Please Select One--", "Children & Teens","Young Adults","Middle-Aged Adults","Senior Citizen"], index=0)
    st.write("Children & Teens (below 20 Years), Young Adults (20-40 Years),  Middle-Aged Adults (41-60 Years),  Senior Citizen (above 60 Years)")
    gender = st.selectbox("Gender", ["--Please Select One--", "Male", "Female"], index=0)
    st.header("B. Clinical Measurements")
    weight = st.slider("Weight (kg)", min_value=0.0, max_value=130.0, value=67.0, step=0.1)
    height = st.slider("Height (cm)", min_value=0.0, max_value=270.0, value=167.0, step=1.0)
    st.header("C. Medical History")
    hypertension = st.selectbox("Hypertension (Select True if patient has hypertension)", ["--Please Select One--", False, True], index=0)
    heart_disease = st.selectbox("Heart Disease (Select True if patient has heart disease)", ["--Please Select One--", False, True], index=0)
    diabetes = st.selectbox("Diabetes (Select True if patient has diabetes)", ["--Please Select One--", False, True], index=0)
    submit = st.form_submit_button("Run Risk Assessment Tool", type="primary", use_container_width=True)

st.divider()

# --- BMI & Obesity Calculation ---
if submit:
    if height > 0 and weight > 0:
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        if bmi >= 30:
            obesity = "Yes"
        else:
            obesity = "No"

        st.subheader("D. Body Mass Index (for information)")
        st.write(f"**BMI**: {bmi:.1f}")
        st.write(f"**Obesity Status**: {obesity}")
        st.caption("Based on World Health Organisation (WHO) guidelines, BMI of over 30 is considered obese")
    else:
        st.warning("Please enter valid weight and height values.")
        st.stop()

st.divider()        

if submit:
    # Prepare input data
    data = {
        "gender": gender,
        "age_category": age_cat,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "diabetes": diabetes,
        "obesity": obesity,
    }
    input_df = pd.DataFrame([data])

    # One-hot encode the input
    input_encoded = pd.get_dummies(input_df)

    # Align with model's training features
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Check if user input matches with model input
#    st.write("Input before encoding:", input_df)
#    st.write("Input after encoding:", input_encoded)
#    st.write("Model expects columns:", model_columns)

# Predict
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    st.success("**Assessment Complete!**")

#   st.write(f"Probability of High Risk: **{proba:.2%}**")

# Results card
    st.subheader("E. Stroke Risk Assessment Summary")
    if proba >= 0.3:  # High risk threshold (adjust as needed)
            st.error(f"**RESULT**: 🚨 **AT RISK**")
            
            # --- CLINICAL RECOMMENDATIONS (ONLY FOR AT RISK) ---
            st.subheader("Clinical Recommendation & Health Tips", divider="red")
            
            # Age-specific recommendations
            if "Senior" in age_cat or "Middle-Aged" in age_cat:
                st.write("• For selected age category: Consider enrolling in **Healthier SG** (https://www.healthiersg.gov.sg) with subsidized health screenings and tailored care plans.")
            
            # Condition-specific recommendations
            if hypertension:
                st.write("• For hypertension: **Regular blood pressure monitoring** and follow-ups with a dedicated GP.")
            if diabetes:
                st.write("• For diabetes: **Regular HbA1c tests & foot/eye screenings** with subsidized glucometers & test strips (if eligible).")
            if obesity == "Yes":
                st.write("• For obesity: **Join LOSE weight program by HPB** with subsidized dietitian consultations and customized exercise plans.")
            if heart_disease:
                st.write("• For heart disease: **Regular lipid profile tests** and free cardiovascular risk assessments.")

    else:
        st.write("**RESULT**: ✅**LOW RISK**")
        
