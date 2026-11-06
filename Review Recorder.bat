@echo off
cd /d "%~dp0"
if exist .venv\Scripts\pythonw.exe (start "" .venv\Scripts\pythonw.exe review_recorder.py) else (start "" pythonw review_recorder.py)
