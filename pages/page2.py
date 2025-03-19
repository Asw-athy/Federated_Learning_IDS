from navigation import make_sidebar
import streamlit as st
from time import sleep
import subprocess

# Define the slow_text function
def slow_text(text, delay=1):
    st.text(text)
    sleep(delay)

make_sidebar()

st.markdown("# Model Training Control")

# Display setup messages
slow_text(" ")
slow_text("Setting up federated learning server...")
slow_text("Initializing clients...")
slow_text("Client 1 setup")
slow_text("Client 2 setup")
slow_text("Client 3 setup")

# Function to run flp.py script
def run_flp_script():
    subprocess.Popen(["start", "cmd", "/k", "python", r"C:/Users/ASWATHY.I/Desktop/MyFLIDS/flp.py"], shell=True)

# Button to open terminal window and run flp.py script
if st.button("Start the Federated learning process"):
    run_flp_script()
