@echo off
set "FFMPEG_BIN=C:\Users\domin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
set "PATH=%FFMPEG_BIN%;%PATH%"
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" main.py
1