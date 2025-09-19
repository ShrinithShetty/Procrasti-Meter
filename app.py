import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Load Model and Columns ---
try:
    model = joblib.load('procrastination_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found! Please run `train_and_save_model.py` first.")
    st.stop()

# --- App Structure ---
st.set_page_config(page_title="Procrastination Predictor", page_icon="🤔")

st.title("🤔 The Procrastination Predictor")
st.markdown("Enter the details of your task and your current state to predict the likelihood of procrastinating.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Task Details")

# Define task categories based on the original data generator
task_categories = ['Code', 'Study/Research', 'Chore', 'Admin/Emails', 'Health/Fitness', 'Social']
task_category = st.sidebar.selectbox("Task Category", options=task_categories)

est_duration = st.sidebar.number_input("Estimated Duration (minutes)", min_value=5, max_value=360, value=60, step=5)
deadline_date = st.sidebar.date_input("Task Deadline")

st.sidebar.header("Enter Your Current State")
mood_level = st.sidebar.slider("Mood Level (1=Low, 5=High)", 1, 5, 3)
energy_level = st.sidebar.slider("Energy Level (1=Low, 5=High)", 1, 5, 3)
hours_of_sleep = st.sidebar.number_input("Hours of Sleep Last Night", min_value=0.0, max_value=16.0, value=7.0, step=0.5)
perceived_enjoyment = st.sidebar.slider("Perceived Enjoyment (1=Dread it, 5=Love it)", 1, 5, 3)


# --- Prediction Logic ---
if st.sidebar.button("Predict Procrastination Risk"):

    # 1. Get current time information
    now = datetime.now()
    day_of_week = now.weekday()  # Monday is 0, Sunday is 6
    hour_of_day = now.hour
    
    # 2. Calculate deadline proximity
    deadline_datetime = datetime.combine(deadline_date, datetime.min.time())
    deadline_proximity = (deadline_datetime - now).days

    # 3. Create a DataFrame from the inputs
    input_data = {
        'Estimated_Duration_Mins': [est_duration],
        'Mood_Level_1_5': [mood_level],
        'Energy_Level_1_5': [energy_level],
        'Hours_of_Sleep': [hours_of_sleep],
        'Perceived_Enjoyment_1_5': [perceived_enjoyment],
        'Day_of_Week': [day_of_week],
        'Hour_of_Day': [hour_of_day],
        'Deadline_Proximity_Days': [deadline_proximity],
        'Task_Category': [task_category] # Keep as category for one-hot encoding
    }
    input_df_raw = pd.DataFrame(input_data)

    # 4. One-Hot Encode the input data to match the model's training format
    input_df_encoded = pd.get_dummies(input_df_raw)
    
    # 5. Align columns with the model's training columns
    # This is a crucial step to ensure the prediction works correctly
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, input_df_encoded])
    final_df = final_df.fillna(0) # Fill any missing columns with 0
    final_df = final_df[model_columns] # Ensure column order is identical

    # 6. Make prediction
    prediction = model.predict(final_df)[0]
    prediction_proba = model.predict_proba(final_df)[0]

    # --- Display Results ---
    st.header("Prediction Result")

    if prediction == 1:
        st.error("🚨 High Risk of Procrastination")
        st.write(f"The model predicts a **{prediction_proba[1]*100:.2f}% probability** that you will delay this task.")
        st.markdown("#### Suggestions:")
        st.markdown("- **Break it down:** Can you split this task into smaller, 15-minute actions?")
        st.markdown("- **Just 5 minutes:** Commit to working on it for just five minutes. You might find it easier to continue once you start.")
        st.markdown("- **Change your environment:** Move to a different location, like a library or coffee shop.")
    else:
        st.success("✅ Low Risk - You're likely to start on time!")
        st.write(f"The model predicts a **{prediction_proba[0]*100:.2f}% probability** that you will get this done without delay.")
        st.balloons()
        st.markdown("#### Great job! Seize the momentum and get started.")
