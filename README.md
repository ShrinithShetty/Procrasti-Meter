🤔 The Procrastination Predictor
This is an end-to-end data science project that trains a machine learning model to predict the likelihood of procrastination based on task details and personal state. The project culminates in a simple web application built with Streamlit for real-time predictions.

🚀 Project Overview
The core idea is to use personal behavioral data to understand the triggers of procrastination. The model, a Random Forest Classifier, learns from a synthetic dataset designed to mimic the habits of a student. It identifies patterns in how task enjoyment, energy levels, sleep, and deadlines influence the decision to delay a task.

The final model is deployed in a Streamlit web app that provides a "procrastination risk score" for new tasks.

Key Features:
Data Generation: A Python script to create a large, realistic synthetic dataset.

Feature Engineering: Creation of predictive features like Deadline_Proximity_Days and a binary Procrastinated target variable.

Handling Class Imbalance: Use of SMOTE (Synthetic Minority Over-sampling Technique) to create a fair and balanced model.

Model Training: Training a Random Forest Classifier and saving the final artifact using joblib.

Interactive Web App: A user-friendly interface built with Streamlit to interact with the trained model.

🔧 How to Run This Project
Follow these steps to get the project running on your local machine.

1. Prerequisites
Python 3.8+

Anaconda or Miniconda (recommended for managing environments)

2. Clone the Repository
git clone <your-repository-url>
cd <repository-name>

3. Set Up the Environment & Install Dependencies
It's highly recommended to use a virtual environment.

# Create and activate a conda environment
conda create --name procrastination_env python=3.10 -y
conda activate procrastination_env

# Install the required packages
pip install -r requirements.txt

4. Generate the Dataset
If you don't have the procrastination_big_data_v2.csv file, you'll need to generate it first.
(Note: You will need to have the data generation script from our previous steps saved as generate_data.py)

python generate_data.py

5. Train and Save the Model
Run the training script. This will create the procrastination_model.joblib and model_columns.joblib files.

python train_and_save_model.py

6. Launch the Streamlit App
You're all set! Run the app script to launch the web application.

streamlit run app.py

Your web browser will open, and you can start making predictions!

📁 Project Structure
.
├── 📄 .gitignore              # Files to be ignored by Git
├── 🚀 app.py                   # The Streamlit web application script
├── 📜 README.md                # This readme file
├── 📋 requirements.txt         # Project dependencies
├── 🧠 train_and_save_model.py    # Script to train and save the ML model
└── (Other generated files)
    ├── 📊 procrastination_big_data_v2.csv  (Ignored by Git)
    ├── 🤖 procrastination_model.joblib     (Ignored by Git)
    └── 🏛️ model_columns.joblib           (Ignored by Git)
