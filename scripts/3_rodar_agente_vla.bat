@echo off
chcp 65001 >nul
title [3] Agente VLA Autonomo (Mineflayer - sem DirectInput)
cd /d "%~dp0.."

echo ================================================================
echo   [3] Agente VLA Autonomo - Nova Arquitetura Mineflayer
echo   Visao  : GET /frame - raycaster voxel (1a pessoa, ~10ms/frame)
echo   Acoes  : POST /action -> bot.setControlState()
echo   Reward : GET /delta   -> deslocamento real no jogo
echo   Dataset: dataset/live_trajectories.jsonl
echo ================================================================
echo.
echo IMPORTANTE: Inicie o servidor Mineflayer [2] primeiro.
echo             Aguarde aparecer "[Frame] Pronto em ..."
echo             antes de iniciar o agente.
echo.
echo Pressione qualquer tecla para iniciar...
pause >nul

python run_vla_agent.py

pause
