import os
import csv
import matplotlib.pyplot as plt

def plot_training_loss():
    """
    Reads loss_history.csv, plots a clean line chart using matplotlib,
    saves it as loss_chart.png, and displays it on screen.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "loss_history.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print("Please train the model first using train.py to generate loss history.")
        return
        
    epochs = []
    losses = []
    
    print(f"Reading loss history from {csv_path}...")
    try:
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for row in reader:
                if row:
                    epochs.append(int(row[0]))
                    losses.append(float(row[1]))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    if not epochs:
        print("Error: No training data found in loss_history.csv.")
        return
        
    print("Plotting training loss chart...")
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title("Model Training Loss History", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the chart as loss_chart.png
    output_image = os.path.join(base_dir, "loss_chart.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Saved loss chart to {output_image}")
    
    # Display it on screen
    print("Displaying chart on screen. Close the window to continue.")
    plt.show()

if __name__ == '__main__':
    plot_training_loss()
