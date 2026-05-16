import subprocess
import sys
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    header = """
=================================================
             COSMO MUSIC AI
=================================================
"""
    print(header)

def run_script(command, script_name):
    print(f"\nRunning {script_name}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print("\nDone!")
    except subprocess.CalledProcessError:
        print(f"\nError: The script '{script_name}' failed to execute properly.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

def main():
    while True:
        clear_screen()
        print_header()
        print("  [1] Preprocess MIDI data (run preprocess.py)")
        print("  [2] Train the model (run train.py)")
        print("  [3] Generate music (run generate.py)")
        print("  [4] Plot training loss (run plot_loss.py)")
        print("  [5] Launch web UI (run: streamlit run app.py)")
        print("  [6] Exit")
        print("=================================================")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            run_script(f"{sys.executable} preprocess.py", "preprocess.py")
        elif choice == '2':
            run_script(f"{sys.executable} train.py", "train.py")
        elif choice == '3':
            run_script(f"{sys.executable} generate.py", "generate.py")
        elif choice == '4':
            run_script(f"{sys.executable} plot_loss.py", "plot_loss.py")
        elif choice == '5':
            run_script(f"{sys.executable} -m streamlit run app.py", "app.py")
        elif choice == '6':
            print("\nExiting Cosmo Music AI. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid input! Please enter a number between 1 and 6.")
            
        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting Cosmo Music AI. Goodbye!")
        sys.exit(0)
