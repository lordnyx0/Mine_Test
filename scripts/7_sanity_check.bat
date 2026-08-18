@echo off
chcp 65001 >nul
title [7] Sanity Check VLA (Verificacao de Saude)
cd /d "%~dp0.."

echo ================================================================
echo   [7] Sanity Check - Verificacao do Modelo VLA
echo   - Testa formas dos tensores
echo   - Verifica ausencia de NaN
echo   - Mede uso de VRAM
echo ================================================================
echo.

python sanity_check_vla.py

pause
