@echo off
chcp 65001 >nul
title [5] Gerar Dataset de Locomocao Sintetico
cd /d "%~dp0.."

echo ================================================================
echo   [5] Gerador de Dataset de Locomocao Sintetico
echo   - Gera 320 amostras visuais de navegacao (W/S/A/D + Camera)
echo   - Salva em dataset/locomotion_trajectories.jsonl
echo ================================================================
echo.

python generate_locomotion_dataset.py

echo.
echo Dataset gerado com sucesso!
pause
