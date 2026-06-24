---
trigger: always_on
---

# Windows & uv Execution Protocol

## Terminal Protocol
- **CRITICAL:** Always use `cmd /c` for all shell executions to ensure the process terminates correctly and sends an EOF signal.
- Avoid interactive shells completely. Example: Use `cmd /c pip list`, not `pip list`.

## Python Environment
- The active virtual environment is located in the `.venv` subdirectory.
- Always default to using this project's local Python via '.venv\Scripts\python.exe'
- When executing a script or inspecting data (like reading a CSV), always use `cmd /c .venv\Scripts\python.exe <script_name>`. Do not rely on the global system Python.