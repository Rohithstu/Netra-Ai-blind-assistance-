@echo off
:: Netra AI — Iriun Wi-Fi Fix (Robust Batch Version)
:: This script fixes firewall, network category, and discovery services.

title Netra AI - Iriun Connectivity Fix
echo ======================================================
echo Netra AI: Fixing Iriun Webcam Connectivity
echo ======================================================
echo.

:: Check for Administrator Privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :RunFix
) else (
    echo [IMPORTANT] This fix requires Administrative permissions.
    echo Requesting elevation...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:RunFix
echo [1/4] Adding AGGRESSIVE Firewall rules for Iriun...
powershell -Command "Remove-NetFirewallRule -DisplayName 'Iriun Webcam (Inbound)' -ErrorAction SilentlyContinue; New-NetFirewallRule -DisplayName 'Iriun Webcam (Inbound)' -Direction Inbound -Action Allow -Program 'Any' -LocalPort 49152-65535 -Protocol UDP -Profile Any -Description 'Allow Iriun High Ports'; New-NetFirewallRule -DisplayName 'Iriun Webcam (App)' -Direction Inbound -Action Allow -Program 'C:\Program Files\Iriun Webcam\IriunWebcam.exe' -Profile Any; Write-Host '✔ Firewall rules applied.' -ForegroundColor Green"

echo.
echo [2/4] Restarting Iriun Hub Service...
powershell -Command "Stop-Process -Name 'IriunWebcam' -ErrorAction SilentlyContinue; Start-Sleep -s 1; Write-Host '✔ Service reset requested.' -ForegroundColor Green"

echo.
echo [3/4] Enabling Discovery (UPnP/SSDP)...
powershell -Command "Set-Service -Name 'SSDPSRV' -StartupType Automatic; Start-Service -Name 'SSDPSRV' -ErrorAction SilentlyContinue; Set-Service -Name 'upnphost' -StartupType Automatic; Start-Service -Name 'upnphost' -ErrorAction SilentlyContinue; Write-Host '✔ Discovery ready.' -ForegroundColor Green"

echo.
echo [4/4] Forcing Wi-Fi to 'Private' Mode...
powershell -Command "$p = Get-NetConnectionProfile -IPv4Connectivity Internet; if ($p) { Set-NetConnectionProfile -InterfaceIndex $p.InterfaceIndex -NetworkCategory Private; Write-Host '✔ Network set to Private.' -ForegroundColor Green } else { Write-Host '⚠ No active Internet Wi-Fi found.' -ForegroundColor Yellow }"

echo.
echo [Bonus] Flushing DNS Cache...
ipconfig /flushdns >nul

echo.
echo ======================================================
echo FIX COMPLETE!
echo ======================================================
echo 1. RESTART the Iriun Desktop app on your PC.
echo 2. OPEN the Iriun app on your phone.
echo.
echo If they still don't connect, make sure BOTH are on the
echo SAME Wi-Fi network.
echo.
pause
