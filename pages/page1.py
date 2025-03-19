from navigation import make_sidebar
import streamlit as st
from time import sleep
from digital_twin import DigitalTwin
from loader import DataLoader
import pandas as pd

make_sidebar()

# Initialize Digital Twin
dt = DigitalTwin(client_id="client1", loc=50, scale=10, threshold=20)

st.markdown("# Data Input")

# File uploaders
uploaded_file1 = st.file_uploader("Upload the Training Dataset", type=['csv', 'xlsx'], key="file1")
uploaded_file2 = st.file_uploader("Upload the Testing Dataset", type=['csv', 'xlsx'], key="file2")

# Train button
if uploaded_file1 and uploaded_file2:
    if st.button("Train", type="primary"):
        try:
            # Read uploaded files directly
            train_data = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith('.csv') else pd.read_excel(uploaded_file1)
            test_data = pd.read_csv(uploaded_file2) if uploaded_file2.name.endswith('.csv') else pd.read_excel(uploaded_file2)

            # Create DataLoader object with the data (not file paths)
            data_loader = DataLoader(train_file_path=None, test_file_path=None)
            X_train, Y_train, X_test, Y_test = data_loader.preprocess_data(train_data, test_data)

            # Align Digital Twin's behavior
            dt.update_normal_behavior(X_train)

            # Validate the aligned training data
            is_valid_train = dt.validate_data(X_train)

            if not is_valid_train:
                st.error("Training data validation failed!")
                st.stop()

            st.success("Training data validated by Digital Twin after alignment!")
            st.info("Training started...")
            sleep(0.5)
            st.switch_page("pages/page2.py")

        except ValueError as e:
            st.error(f"Error occurred: {str(e)}")







