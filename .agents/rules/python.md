---
trigger: always_on
---

# Windows & uv Execution Protocol

## Terminal Protocol
- **CRITICAL:** Always use `cmd /c` for all shell executions to ensure the process terminates correctly and sends an EOF signal.
- Avoid interactive shells completely. Example: Use `cmd /c pip list`, not `pip list`.

## Python Environment
- The active virtual environment is located in the original LSTM project directory:
  `C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\.venv` (or relative `..\LSTM_hunt_label\.venv`).
- Always default to using Python via `..\LSTM_hunt_label\.venv\Scripts\python.exe` (or `C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\.venv\Scripts\python.exe`).
- When executing a script or inspecting data, use:
  `cmd /c ..\LSTM_hunt_label\.venv\Scripts\python.exe <script_name>`. Do not rely on the global system Python.