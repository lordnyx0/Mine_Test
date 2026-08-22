@echo off
chcp 65001 >nul
title [Simulador Minecraft Offline - 8 Ambientes]
cd /d "%~dp0..\.."

echo ================================================================
echo   🎮 SIMULADOR MINECRAFT OFFLINE (8 AMBIENTES PARALELOS)
echo   Porta HTTP : 3002
echo   Endpoint   : http://127.0.0.1:3002/lote/info
echo ================================================================
echo.

node servidor_offline.js

pause
