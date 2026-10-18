@echo off
setlocal
rem ===================================================================
rem  ICESat Photon Profile Viewer - launcher  (ASCII only on purpose:
rem  cmd.exe parses .bat files with the OEM code page, so non-ASCII
rem  comments here would be garbled on non-UTF8 systems.)
rem
rem  Prefers the conda env "waterenv" (already contains numpy,
rem  matplotlib, h5py, Pillow, requests, pyproj, scipy, pandas);
rem  falls back to "python" on PATH.
rem
rem  Usage:
rem      run.bat                          start the viewer
rem      run.bat --demo                   create + open demo data
rem      run.bat --open "D:\data\ATL03_xxx.h5"
rem      run.bat --provider esri_img      key-free basemap
rem      run.bat --token YOUR_TIANDITU_TK
rem ===================================================================

set "PYEXE="
if exist "%USERPROFILE%\miniconda3\envs\waterenv\python.exe" set "PYEXE=%USERPROFILE%\miniconda3\envs\waterenv\python.exe"
if not defined PYEXE if exist "%USERPROFILE%\anaconda3\envs\waterenv\python.exe" set "PYEXE=%USERPROFILE%\anaconda3\envs\waterenv\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\miniconda3\envs\waterenv\python.exe" set "PYEXE=%LOCALAPPDATA%\miniconda3\envs\waterenv\python.exe"
if not defined PYEXE if exist "D:\ProgramData\miniconda3\envs\waterenv\python.exe" set "PYEXE=D:\ProgramData\miniconda3\envs\waterenv\python.exe"
if not defined PYEXE if exist "C:\ProgramData\miniconda3\envs\waterenv\python.exe" set "PYEXE=C:\ProgramData\miniconda3\envs\waterenv\python.exe"

if not defined PYEXE (
    echo [INFO] conda env "waterenv" not found - falling back to python on PATH
    set "PYEXE=python"
)

echo [INFO] interpreter: %PYEXE%
"%PYEXE%" "%~dp0main.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] the viewer exited with code %RC%
    pause
)
endlocal & exit /b %RC%
