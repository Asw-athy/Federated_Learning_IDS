import numpy as np
import matplotlib.pyplot as plt
import os
import re


def moving_average(data, window_size=3):
    """Applies a moving average filter to smooth fluctuations."""
    if len(data) < window_size:
        return data  # Return original data if not enough points for smoothing
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def extract_metrics(file_path):
    """Reads and extracts LOSS and MODEL ACCURACY from the given output file."""
    loss_data = {}
    model_accuracy = {}

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract the LOSS section
        loss_section = re.search(r"'LOSS':([\s\S]*?)'MODEL ACCURACY':", content)

        # Extract the MODEL ACCURACY section (corrected from 'BINARY ACCURACY')
        acc_section = re.search(r"'MODEL ACCURACY':([\s\S]*?)$", content)

        if loss_section:
            loss_matches = re.findall(r"Round (\d+): ([\d.]+)", loss_section.group(1))
            loss_data = {int(round_num): float(loss_value) for round_num, loss_value in loss_matches}

        if acc_section:
            acc_matches = re.findall(r"Round (\d+): ([\d.]+)", acc_section.group(1))
            model_accuracy = {int(round_num): float(acc_value) for round_num, acc_value in acc_matches}

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return loss_data, model_accuracy


def plot_metrics(file_path, save_path=None, show_plot=True):
    """Reads the metrics from the output file and generates a plot."""
    loss_data, model_accuracy = extract_metrics(file_path)

    if not loss_data or not model_accuracy:
        print("[ERROR] No valid data extracted for plotting.")
        return

    # Sort data by round
    rounds = sorted(loss_data.keys())
    loss_values = [loss_data[r] for r in rounds]
    accuracy_values = [model_accuracy[r] for r in rounds]

    # Apply Moving Average to Loss (window size = 3)
    smoothed_loss = moving_average(loss_values, window_size=3)

    # Adjust x-axis values for smoothed loss (since moving average reduces length)
    smoothed_rounds = rounds[:len(smoothed_loss)]

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Plot Original Loss (Faded for Reference)
    plt.plot(rounds, loss_values, 'r--o', label='Loss')

    # Plot Smoothed Loss
    # plt.plot(smoothed_rounds, smoothed_loss, 'r-', linewidth=2, label='Smoothed Loss')

    # Plot Model Accuracy
    plt.plot(rounds, accuracy_values, 'g--o', label='Model Accuracy')

    # Labels and Titles
    plt.title('Training Metrics Over Rounds', fontsize=14)
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Binary Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    # Show the plot if enabled
    if show_plot:
        plt.show()
    else:
        plt.close()


# Example usage:
if __name__ == "__main__":
    directory_path = r"C:/Users/ASWATHY.I/Desktop/MyFLIDS"
    filename = "output.txt"
    file_path = os.path.join(directory_path, filename)

    # Define where to save the plot
    save_path = os.path.join(directory_path, "plot.png")

    # Call the function to generate the plot
    plot_metrics(file_path, save_path=save_path, show_plot=True)
