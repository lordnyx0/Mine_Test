@echo off
chcp 65001 >nul
title [8] Interface de Treinamento por Feedback Humano
cd /d "%~dp0.."

echo ================================================================
echo   [8] Interface de Feedback Humano (trainer_interface.py)
echo   - Loop de chat com o agente via llama-server (porta 8085)
echo   - !+ para recompensa positiva
echo   - !- para recompensa negativa
echo ================================================================
echo.
echo IMPORTANTE: Inicie o servidor LLM [1] antes de usar esta interface.
echo.

python trainer_interface.py

pause
