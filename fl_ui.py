import streamlit as st
import os

# Function to read the output file
def read_output_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Set up Streamlit UI
def display_ui():
    st.title("Federated Learning - Results Dashboard")

    # Define the directory and file name
    directory_path = r"C:/Users/ASWATHY.I/Desktop/MyFLIDS"  # Ensure this path is correct
    filename = "output.txt"
    file_path = os.path.join(directory_path, filename)

    # Display the summary output or a message if the file doesn't exist
    if os.path.exists(file_path):
        st.subheader("Federated Learning Summary")
        summary_output = read_output_file(file_path)
        st.text_area("FL Summary Output", summary_output, height=300)
        
        # Provide an option to download the summary as a file
        with open(file_path, 'r') as f:
            st.download_button(
                label="Download Summary Output",
                data=f,
                file_name=filename,
                mime="text/plain"
            )
    else:
        st.warning("The output file `output.txt` does not exist. Please ensure the server has run successfully.")

    # Optional: Button to trigger reloading the output file (e.g., after a new federated learning run)
    if st.button("Reload Output"):
        if os.path.exists(file_path):
            st.experimental_rerun()
        else:
            st.warning("Output file not found. Please run the server to generate the output.")

if __name__ == "__main__":
    display_ui()
