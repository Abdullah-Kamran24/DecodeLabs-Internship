@echo off
REM ============================================================
REM  Rule-Based AI Chatbot - Windows Launcher
REM  Double-click this file, or run "run.bat" from cmd / PowerShell,
REM  to start the chatbot without typing the python command yourself.
REM ============================================================

echo Starting Nova - Rule-Based AI Chatbot...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo [!] Something went wrong. Make sure Python 3.8+ is installed
    echo     and added to your PATH, then try again.
    echo     Check with:  python --version
)

echo.
pause
