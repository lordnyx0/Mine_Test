@echo off
chcp 65001 >nul
title [Servidor LLM GGUF Q8] - Qwen3Loop Local
cd /d "%~dp0..\.."

set "SERVER_EXE=C:\Users\Nyx\.unsloth\.staging\llama.cpp.staging-eu_6bjrp\build-qwen3loop\bin\Release\llama-server.exe"

:: Permite passar o modelo por argumento ou usa o base professor como padrão
set "MODELO_PADRAO=%~dp0..\modelos\base_professor_q8_0.gguf"
if not "%~1"=="" set "MODELO_PADRAO=%~1"

echo ================================================================
echo   🚀 SERVIDOR LLM LOCAL (COMPATÍVEL OPENAI / HTTP REST)
echo   Modelo : %MODELO_PADRAO%
echo   Porta  : 8085
echo   Host   : 127.0.0.1
echo   GPU Offload: Total (-ngl 99)
echo ================================================================
echo.

if not exist "%SERVER_EXE%" (
    echo [ERRO] Executável llama-server.exe não encontrado em:
    echo        %SERVER_EXE%
    pause
    exit /b 1
)

if not exist "%MODELO_PADRAO%" (
    echo [AVISO] Modelo não encontrado em: %MODELO_PADRAO%
    echo Verifique os arquivos na pasta 'modelos'.
    pause
    exit /b 1
)

"%SERVER_EXE%" -m "%MODELO_PADRAO%" -ngl 99 -c 2048 --port 8085 --host 127.0.0.1 -np 1 --no-webui

pause
