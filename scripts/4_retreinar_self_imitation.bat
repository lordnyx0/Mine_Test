@echo off
chcp 65001 >nul
title [4] Self-Imitation RL Retraining
cd /d "%~dp0.."

echo ================================================================
echo   [4] Self-Imitation RL Fine-Tuning
echo   - Le trajetorias positivas de dataset/live_trajectories.jsonl
echo   - Treina apenas nas acoes com reward ^> 0.0
echo   - Atualiza checkpoints_vla/vla_locomotion.pt
echo ================================================================
echo.
echo Pressione qualquer tecla para iniciar o retreinamento...
pause >nul

python retrain_self_imitation.py

echo.
echo Fine-tuning concluido! Checkpoint atualizado.
pause
