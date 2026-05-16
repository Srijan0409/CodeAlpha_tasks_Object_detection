@echo off
echo Installing Music Generation AI -- please wait...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Make sure Python 3.9-3.12 is installed and added to PATH
) else (
    echo Done! Now run: python run.py
)
pause
