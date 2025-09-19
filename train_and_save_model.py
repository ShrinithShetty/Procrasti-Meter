import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

print("Starting model training process...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('procrastination_big_data_v2.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'procrastination_big_data_v2.csv' not found.")
    print("Please run the data generation script first.")
    exit()

# --- 2. Feature Engineering ---
# Convert date/time columns to datetime objects
df['Actual_Start_Time'] = pd.to_datetime(df['Actual_Start_Time'])
df['Planned_Start_Time'] = pd.to_datetime(df['Planned_Start_Time'])
df['Deadline_Date'] = pd.to_datetime(df['Deadline_Date'])

# Create the target variable
df['Delay_Minutes'] = (df['Actual_Start_Time'] - df['Planned_Start_Time']).dt.total_seconds() / 60
df['Procrastinated'] = (df['Delay_Minutes'] > 30).astype(int)

# Create time-based & deadline features
df['Day_of_Week'] = df['Actual_Start_Time'].dt.dayofweek
df['Hour_of_Day'] = df['Actual_Start_Time'].dt.hour
df['Deadline_Proximity_Days'] = (df['Deadline_Date'] - df['Actual_Start_Time']).dt.days

print("Feature engineering complete.")

# --- 3. Prepare Data for Modeling ---
y = df['Procrastinated']
features_to_include = [
    'Estimated_Duration_Mins', 'Mood_Level_1_5', 'Energy_Level_1_5', 
    'Hours_of_Sleep', 'Perceived_Enjoyment_1_5', 'Day_of_Week', 
    'Hour_of_Day', 'Deadline_Proximity_Days', 'Task_Category'
]
X_raw = df[features_to_include]
X = pd.get_dummies(X_raw, columns=['Task_Category'], drop_first=True)

# --- 4. Handle Class Imbalance with SMOTE ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("SMOTE applied to balance the dataset.")

# --- 5. Train the Final Model ---
# We use the full, resampled dataset for the final production model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_resampled, y_resampled)
print("Final Random Forest model trained successfully.")

# --- 6. Save the Model and Columns ---
# Save the trained model
joblib.dump(final_model, 'procrastination_model.joblib')

# Save the list of columns the model was trained on
joblib.dump(X.columns.tolist(), 'model_columns.joblib')

print("\nModel and column list have been saved to disk:")
print("- procrastination_model.joblib")
print("- model_columns.joblib")
