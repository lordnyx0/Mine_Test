@echo off
chcp 65001 >nul
title MINECRAFT VLA - Iniciar Tudo

set "BASE_DIR=%~dp0.."
set "SCRIPTS_DIR=%~dp0"
cd /d "%BASE_DIR%"

echo ================================================================
echo   INICIAR TUDO - Mineflayer VLA Agent Server
echo ================================================================
echo.

echo [1/3] Descobrindo porta LAN do Minecraft (ate 8s)...
python "%SCRIPTS_DIR%discover_lan.py"
if %errorlevel% neq 0 (
    echo ERRO ao descobrir porta LAN.
    pause
    exit /b 1
)

:: Carrega MC_HOST e MC_PORT do arquivo gerado
call "%SCRIPTS_DIR%_lan_vars.bat"
del "%SCRIPTS_DIR%_lan_vars.bat" >nul 2>&1

echo.
echo Host : %MC_HOST%
echo Porta: %MC_PORT%
echo.

echo [2/3] Abrindo Mineflayer em nova janela...
start "Mineflayer VLA Server" cmd /k "chcp 65001 >nul && title Mineflayer VLA Server && set MC_HOST=%MC_HOST%&& set MC_PORT=%MC_PORT%&& set HTTP_PORT=3001&& set VIEW_PORT=3002&& cd /d "%BASE_DIR%\mineflayer_server" && node state_server.js"

echo.
echo [3/3] Aguardando bot spawnar e viewer carregar (15s)...
echo       Acompanhe a janela do Mineflayer ao lado.
python -c "import time; [print(f'  {15-i}s...', flush=True) or time.sleep(1) for i in range(15)]"

echo.
echo Iniciando Agente VLA Autonomo...
echo Ctrl+C nesta janela para parar.
echo.
python "%BASE_DIR%\run_vla_agent.py"

echo.
echo Agente encerrado.
pause
