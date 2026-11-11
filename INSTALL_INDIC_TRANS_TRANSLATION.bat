@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo SpeechMind - IndicTrans2 Windows compatibility check
echo ==========================================================
echo.
echo STOP: IndicTransToolkit 1.1.1 builds a native Cython extension.
echo On this PC, pip failed because Microsoft C++ Build Tools (MSVC 14+)
echo are not installed or are not available to this terminal.
echo.
echo This launcher intentionally does NOT retry pip installation.
echo The main .venv and .venv_srota environments are left unchanged.
echo Do not install random compiler packages or rerun this launcher.
echo.
echo Translation remains disabled until we provide a Windows-compatible
echo runtime or move only the translation worker to a supported Linux/WSL
echo environment and verify it end-to-end.
echo.
pause
exit /b 2
