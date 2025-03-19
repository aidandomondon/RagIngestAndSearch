@echo off

REM 1) Create the virtual environment
python -m venv .venv

REM 2) Activate the environment (note: "call" is needed so the script continues after activation)
call .venv\Scripts\activate.bat

REM 3) Install dependencies
pip install -r requirements.txt

REM 4) Run setup.py (if needed)
python setup.py

