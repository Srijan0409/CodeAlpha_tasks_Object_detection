# Cosmo Music AI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An LSTM-based deep learning system that composes original music from MIDI training data, with a space-themed Streamlit web interface.

## Demo

![App Preview](placeholder.gif)
*(replace with screenshot of app.py)*

## Features

* ✅ Trains on real MIDI music files (classical, jazz, ambient)
* ✅ LSTM neural network with 512 units per layer
* ✅ Temperature-controlled creativity slider
* ✅ Space-themed animated Streamlit web UI
* ✅ Real-time training loss chart
* ✅ One-click MIDI and WAV download
* ✅ Beginner-friendly — no music knowledge needed

## How it works

**1. Data Processing**
The system first reads raw MIDI files placed in the `midi_data` folder. It extracts the musical notes and chords from these files, mapping each unique sound to a numeric integer so the neural network can understand them.

**2. Sequence Generation**
The numeric notes are grouped into fixed-length sequences. The network is trained to look at a sequence of notes (for instance, the first 100 notes of a song) and predict the single note that should come next.

**3. Model Training**
An LSTM (Long Short-Term Memory) neural network processes these sequences. LSTMs are specifically designed to remember long-term patterns, making them ideal for learning the structure, rhythm, and melody of the training music over multiple epochs.

**4. Music Generation**
Once trained, the model is given a random starting sequence or "seed". It predicts the next note, appends it to the sequence, and repeats this process to generate an entirely new stream of music based on the learned patterns.

**5. Output Conversion**
Finally, the newly generated sequence of numeric integers is converted back into real musical notes and chords. The system saves this output as a standard MIDI file, which can be played on any media player or converted into WAV format.

## Project Structure

```text
music_generation_ai/
│
├── midi_data/              # Folder to place your training MIDI files
├── app.py                  # Streamlit web dashboard and UI
├── generate.py             # Script to generate new music using the trained model
├── model.py                # Defines the LSTM neural network architecture
├── plot_loss.py            # Generates charts to visualize model training progress
├── preprocess.py           # Parses MIDI files and extracts notes/chords
├── requirements.txt        # Python library dependencies
├── run.py                  # Interactive CLI menu to run the whole pipeline
├── setup.bat               # Automated installer script for Windows
├── setup.sh                # Automated installer script for Mac/Linux
├── train.py                # Script to train the neural network
└── README.md               # Project documentation
```

## Quick Start

**Step 1:** Clone or download the project
**Step 2:** Run `setup.bat` (Windows) or `bash setup.sh` (Mac/Linux)
**Step 3:** Add your MIDI files to the `midi_data/` folder
**Step 4:** Run `python run.py` and follow the interactive menu

## Manual Setup

For advanced users who prefer to set up the environment manually:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess the MIDI files
python preprocess.py

# 3. Train the model
python train.py

# 4. Generate new music
python generate.py

# 5. Launch the Web UI
streamlit run app.py
```

## Technologies

| Tool | Version | Purpose |
| --- | --- | --- |
| Python | 3.9+ | Core programming language |
| TensorFlow / Keras | 2.15+ | Deep learning framework for the LSTM model |
| Streamlit | 1.30+ | Building the interactive web user interface |
| Music21 | 9.1+ | Parsing and manipulating MIDI music data |
| Matplotlib | 3.8+ | Plotting and visualizing training loss curves |

## Sample Output

The generated music is output as a `.mid` (MIDI) file. The output typically mimics the style of the training data—for instance, if trained on classical piano pieces, the generated music will feature complex, flowing piano melodies. You can play this `.mid` file using almost any standard media player (like Windows Media Player or VLC), import it into a Digital Audio Workstation (DAW) like FL Studio or Logic Pro to apply high-quality software instruments, or download it directly as a WAV file from the Streamlit web interface.

## About This Project

Built as Task 3 of the Code Alpha Internship Program. Demonstrates applied deep learning for generative AI in music.

## Author & License

Developed for the Code Alpha Internship.
This project is licensed under the MIT License.
