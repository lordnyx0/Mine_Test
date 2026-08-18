@echo off
chcp 65001 >nul
title [6] Treino VLA - 10 Epocas GPU (train_vla.py)
cd /d "%~dp0.."

echo ================================================================
echo   [6] Treino VLA - Qwen3Loop Visuomotor
echo   - Treina SigLIP + Perceiver Resampler + Action Heads
echo   - 10 epocas no dataset de locomocao
echo   - Salva checkpoint em checkpoints_vla/vla_locomotion.pt
echo   - Requer GPU (RTX 3060 ou superior recomendado)
echo ================================================================
echo.
echo AVISO: Este processo pode levar varias horas.
echo Pressione qualquer tecla para iniciar...
pause >nul

python train_vla.py

echo.
echo Treino concluido! Checkpoint salvo.
pause
