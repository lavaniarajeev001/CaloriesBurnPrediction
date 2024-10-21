import streamlit as st
import pandas as pd
import numpy as np
import pickle

def data():
    df = pd.read_csv("gym_members_exercise_tracking.csv")
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 2})
    df["Workout_Type"] = df["Workout_Type"].map({"Strength": 1, "Cardio": 2, "Yoga": 3, "HIIT": 4})

    # Function to remove outliers
    def outlier(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df[(df[column] > Q1 - 1.5 * IQR) & (df[column] < Q3 + 1.5 * IQR)]
        return df_cleaned

    # Apply outlier function
    df = outlier(df, "Weight (kg)")
    df = outlier(df, "Calories_Burned")
    df = outlier(df, "BMI")

    return df
    

def add_sidebar():
    df = data()
    st.header("Attributes")
    slider_label = [
        ("Age", "Age"),
        ("Gender", "Gender"),
        ("Avg_BPM", "Avg_BPM"),
        ("Session_Duration (hours)", "Session_Duration (hours)"),
        ("Fat_Percentage", "Fat_Percentage")
    ]
    
    input_dict = {}
    for label, key in slider_label:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0,
            max_value=int(df[key].max())
        )
    return input_dict
    
def add_prediction(input_data):
    df = data()
    
    # Load the saved model and scalers
    with open("model.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
        
    with open("scaler_X.pkl", "rb") as scaler_X_in:  # Scaler for features
        scaler_X = pickle.load(scaler_X_in)

    with open("scaler_y.pkl", "rb") as scaler_y_in:  # Scaler for target
        scaler_y = pickle.load(scaler_y_in)

    # Convert user input into array
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # Scale the input features using scaler_X
    input_scaled = scaler_X.transform(input_array)
    
    # Make the prediction
    prediction_scaled = classifier.predict(input_scaled)
    
    # The prediction (target) needs to be reshaped before inverse transform
    prediction_reshaped = np.array(prediction_scaled).reshape(-1, 1)  # Reshape to 2D (n_samples, 1)
    
    # Inverse transform to get the original scale for calories burned
    prediction_original = scaler_y.inverse_transform(prediction_reshaped)
    
    # Display the prediction
    st.subheader("Prediction")
    st.write("The total calories burned are:") 
    st.write(f"{prediction_original[0][0]:.2f}")  # Displaying with two decimal places


def main():
    st.set_page_config(
        page_title="Calorie Burn App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Collect user input from sidebar
    input_data = add_sidebar()
    
    # Main container for the app
    with st.container():
        st.title("Calories Burn App")
        st.write("This app is designed for the prediction of calories burned based on the provided attributes.")
        
    # Trigger prediction when the button is clicked
    if st.button("Predict"):
        add_prediction(input_data)

if __name__ == "__main__":
    main()
