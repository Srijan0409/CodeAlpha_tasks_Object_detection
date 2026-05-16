import os
import glob
import pickle
import time
from tqdm import tqdm
from music21 import converter, instrument, note, chord

def get_notes():
    """ 
    Extract all notes and chords from midi files in the ./midi_data directory.
    Saves the extracted notes to notes.pkl.
    """
    notes = []
    
    # Use os.path to ensure paths are portable and relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    midi_data_dir = os.path.join(base_dir, "midi_data")
    
    # Check if directory exists
    if not os.path.exists(midi_data_dir):
        print(f"Directory not found: {midi_data_dir}")
        print("Creating the directory. Please place some .mid files inside it and run again.")
        os.makedirs(midi_data_dir, exist_ok=True)
        return

    midi_path = os.path.join(midi_data_dir, "*.mid")
    files = glob.glob(midi_path)
    
    if len(files) == 0:
        print(f"Error: No MIDI files found in {midi_data_dir}.")
        print("Please place classical or jazz .mid files in the directory and try again.")
        return

    print(f"Found {len(files)} MIDI files. Starting preprocessing...")
    
    for file in tqdm(files, desc="Parsing MIDI files"):
        try:
            # Parse the MIDI file into a music21 stream
            midi = converter.parse(file)
            
            notes_to_parse = None
            
            # Group by instrument if possible
            parts = instrument.partitionByInstrument(midi)
            if parts: # File has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # File has notes in a flat structure
                notes_to_parse = midi.flat.notes
            
            # Extract notes and chords
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    # Append single notes as string (e.g., "C4", "E3")
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    # Append chords by joining note values (e.g., "4.7.11")
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    
        except Exception as e:
            print(f"\nError parsing {file}: {e}")
            
    if not notes:
        print("No valid notes or chords could be extracted from the provided files.")
        return
        
    print(f"\nTotal notes/chords extracted: {len(notes)}")
    
    # Calculate unique notes
    unique_notes = sorted(set(notes))
    print(f"Total number of unique notes found: {len(unique_notes)}")
    
    # Save the notes to a pickle file
    notes_file = os.path.join(base_dir, "notes.pkl")
    with open(notes_file, 'wb') as filepath:
        pickle.dump(notes, filepath)
        
    print(f"\nnotes.pkl saved — {len(unique_notes)} unique notes found across {len(files)} files")

if __name__ == '__main__':
    try:
        start_time = time.time()
        get_notes()
        elapsed = int(time.time() - start_time)
        print(f"Completed in {elapsed // 60} minutes {elapsed % 60} seconds")
    except Exception as e:
        print(f"Something went wrong: {e}. Please check the README for help.")
