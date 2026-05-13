import os
import sys
import subprocess

def run_script(script_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"\nError: Script '{script_name}' not found at {script_path}")
        return
        
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    
    try:
        # Use the current Python interpreter to run the target script
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while running {script_name} (Exit code: {e.returncode})")
    except Exception as e:
        print(f"\nFailed to execute {script_name}: {e}")
        
    print(f"\n{'='*50}")
    print(f"Finished {script_name}")
    print(f"{'='*50}\n")

def main():
    while True:
        print("\n" + "*"*40)
        print("     MUSIC GENERATION AI - CLI MENU     ")
        print("*"*40)
        print("[1] Preprocess MIDI data")
        print("[2] Train model")
        print("[3] Generate music")
        print("[4] Plot training loss")
        print("[5] Exit")
        print("*"*40)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            run_script("preprocess.py")
        elif choice == '2':
            run_script("train.py")
        elif choice == '3':
            run_script("generate.py")
        elif choice == '4':
            run_script("plot_loss.py")
        elif choice == '5':
            print("\nExiting Music Generation AI CLI. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")

if __name__ == '__main__':
    main()
