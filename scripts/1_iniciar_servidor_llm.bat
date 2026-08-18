@echo off
chcp 65001 >nul
title [1] Servidor LLM Qwen3Loop (fase1_loop_q8_0.gguf)
cd /d "%~dp0.."

set "SERVER_EXE=C:\Users\Nyx\.unsloth\.staging\llama.cpp.staging-eu_6bjrp\build-qwen3loop\bin\Release\llama-server.exe"
set "MODEL_PATH=%~dp0..\fase1_loop_q8_0.gguf"

echo ================================================================
echo   [1] Servidor LLM Qwen3Loop 0.6B
echo   Modelo : %MODEL_PATH%
echo   Porta  : 8085
echo ================================================================
echo.

"%SERVER_EXE%" -m "%MODEL_PATH%" -ngl 99 -c 2048 --port 8085 --host 127.0.0.1 -np 1 --no-webui

pause
