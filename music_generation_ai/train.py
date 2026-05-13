import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Import our custom network builder
from model import create_network

def prepare_sequences(notes, n_vocab):
    """ 
    Prepare the input and output sequences used by the Neural Network.
    Uses a sliding window of 100 notes.
    """
    sequence_length = 100
    
    # Get all unique pitch names sorted
    pitchnames = sorted(set(item for item in notes))
    
    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    network_input = []
    network_output = []
    
    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        
        # Map strings to integers
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input values by dividing by the number of unique notes (n_vocab)
    network_input = network_input / float(n_vocab)
    
    # One-hot encode the output labels
    network_output = to_categorical(network_output)
    
    return network_input, network_output

# Custom callback to print training loss and save every 10 epochs
class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, model_path):
        super(TrainingMonitor, self).__init__()
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        # Print training loss after each epoch
        print(f"\nEpoch {epoch + 1} completed. Loss: {logs.get('loss'):.4f}")
        
        # Save a checkpoint after every 10 epochs
        if (epoch + 1) % 10 == 0:
            self.model.save(self.model_path)
            print(f"--> Saved model checkpoint to {self.model_path}")

def train_network():
    """ 
    Train the Neural Network to generate music.
    Loads data, prepares sequences, builds model, and trains.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notes_file = os.path.join(base_dir, "notes.pkl")
    
    # Gracefully handle file-not-found
    if not os.path.exists(notes_file):
        print(f"Error: {notes_file} not found.")
        print("Please run preprocess.py first to extract notes from MIDI files.")
        return
        
    print(f"Loading notes from {notes_file}...")
    with open(notes_file, 'rb') as filepath:
        notes = pickle.load(filepath)
        
    # Calculate vocab size
    n_vocab = len(set(notes))
    print(f"Loaded {len(notes)} total notes. Unique vocab size: {n_vocab}")
    
    print("Preparing sequences using a sliding window of 100 notes...")
    network_input, network_output = prepare_sequences(notes, n_vocab)
    
    print(f"Input shape: {network_input.shape}")
    print(f"Output shape: {network_output.shape}")
    
    # Build model architecture
    model = create_network(n_vocab)
    
    model_path = os.path.join(base_dir, "music_model.h5")
    
    # Use our custom callback
    monitor_callback = TrainingMonitor(model_path)
    
    print("\nStarting training for 100 epochs...")
    history = model.fit(
        network_input, 
        network_output, 
        epochs=100, 
        batch_size=64, 
        callbacks=[monitor_callback],
        verbose=1 # Show progress bar
    )
    
    # Save the final model
    model.save(model_path)
    print(f"\nTraining complete! Final model saved to {model_path}")
    
    # Save loss history to loss_history.csv
    loss_csv_path = os.path.join(base_dir, "loss_history.csv")
    try:
        import csv
        with open(loss_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss"])
            for epoch_idx, loss_val in enumerate(history.history['loss']):
                writer.writerow([epoch_idx + 1, loss_val])
        print(f"Saved loss history to {loss_csv_path}")
    except Exception as e:
        print(f"Error saving loss history: {e}")

if __name__ == '__main__':
    train_network()
