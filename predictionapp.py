import streamlit as st
import pandas as pd
import joblib

# Function to load trained model

def load_model(model_file):
    model = joblib.load(model_file)
    return model

# Function to make predictions
def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Model Prediction App")

    # Upload model file
    st.sidebar.subheader("Upload Model")
    model_file = st.sidebar.file_uploader("Upload Trained Model", type=["pkl"])

    # Upload preprocessed dataset
    st.sidebar.subheader("Upload Preprocessed Dataset")
    dataset_file = st.sidebar.file_uploader("Upload Preprocessed Dataset (CSV)", type=["csv"])

    if model_file and dataset_file:
        # Load model
        model = load_model(model_file)

        # Load preprocessed dataset
        preprocessed_data = pd.read_csv(dataset_file)

        # Show uploaded data
        st.subheader("Uploaded Preprocessed Dataset")
        st.write(preprocessed_data)

        # Select feature columns
        st.subheader("Select Features for Prediction")
        feature_columns = st.multiselect("Select Feature Columns", preprocessed_data.columns)

        if feature_columns:
            # Input values for selected features
            st.subheader("Input Values for Selected Features")
            input_values = {}
            for feature in feature_columns:
                input_values[feature] = st.number_input(f"Enter value for {feature}", min_value=0.0, step=0.01)

            if st.button("Make Prediction"):
                # Create DataFrame from input values
                input_data = pd.DataFrame([input_values])

                # Make prediction
                prediction = make_prediction(model, input_data)
                st.subheader("Prediction Result")
                st.write(prediction)
        else:
            st.warning("Please select at least one feature column.")

if __name__ == "__main__":
    main()

