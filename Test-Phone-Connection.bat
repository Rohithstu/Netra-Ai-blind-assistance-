@echo off
set /p IP="Enter your Phone's IP address (shown in Iriun app): "
echo Testing connection to %IP%...
ping -n 4 %IP%
if %errorLevel% == 0 (
    echo.
    echo ✅ SUCCESS: Your PC can see your phone!
    echo If Iriun still says 'Looking for phone', try restarting the Iriun app on BOTH devices.
) else (
    echo.
    echo ❌ FAILURE: Your PC CANNOT see your phone.
    echo 1. Check if both are on SAME Wi-Fi.
    echo 2. Check if your Wi-Fi is 'Private' (not Public).
    echo 3. Your router might be blocking 'AP Isolation'.
)
pause
