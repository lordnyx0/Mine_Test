@echo off
chcp 65001 >nul
title [2] Mineflayer VLA Agent Server

set "BASE_DIR=%~dp0.."
set "SCRIPTS_DIR=%~dp0"
cd /d "%BASE_DIR%"

echo ================================================================
echo   [2] Mineflayer VLA Agent Server v4
echo   GET  /frame  - visao 1a pessoa do bot (JPEG, raycaster voxel)
echo   POST /action - executa WASD no jogo
echo   GET  /delta  - deslocamento real (recompensa)
echo   GET  /stats  - metricas de renderizacao
echo ================================================================
echo.
echo IMPORTANTE: Minecraft 1.8.9 deve estar aberto em modo LAN.
echo.

echo Descobrindo porta LAN (ate 8s)...
python "%SCRIPTS_DIR%discover_lan.py"
if %errorlevel% neq 0 (
    echo ERRO: Nao foi possivel descobrir a porta LAN.
    pause
    exit /b 1
)

call "%SCRIPTS_DIR%_lan_vars.bat"
del "%SCRIPTS_DIR%_lan_vars.bat" >nul 2>&1

echo.
echo Host : %MC_HOST%
echo Porta: %MC_PORT%
echo.

set HTTP_PORT=3001
set VIEW_PORT=3002

rem Renderizador: "voxel" (padrao, ~10ms/frame) ou "puppeteer" (legado, 80-200ms)
set RENDERER=voxel
rem Resolucao do raycast = FRAME_W/FRAME_SCALE x FRAME_H/FRAME_SCALE
rem   FRAME_SCALE=5 ou 6 deixa mais rapido; 3 deixa mais nitido e mais lento
set FRAME_W=640
set FRAME_H=360
set FRAME_SCALE=4
set FRAME_DIST=64

cd /d "%BASE_DIR%\mineflayer_server"
node state_server.js

pause
