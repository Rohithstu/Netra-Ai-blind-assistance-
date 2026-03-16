# Netra AI — Iriun Wi-Fi Fix Script
# This script automates Windows Firewall rules and Network Discovery to fix Iriun connection issues.

Write-Host "--- Netra AI: Fixing Iriun Webcam Connectivity ---" -ForegroundColor Cyan

# 1. Check for Administrative privileges
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warning "Please RUN THIS SCRIPT AS ADMINISTRATOR to apply firewall changes."
    exit
}

# 2. Add Firewall Rules for Iriun Webcam
Write-Host "[1/3] Adding Windows Firewall rules for Iriun..." -ForegroundColor Yellow
$iriunPath = "C:\Program Files\Iriun Webcam\IriunWebcam.exe"

if (Test-Path $iriunPath) {
    # Remove existing rules to avoid duplicates
    Remove-NetFirewallRule -DisplayName "Iriun Webcam (Inbound)" -ErrorAction SilentlyContinue
    
    # Add new Inbound rule
    New-NetFirewallRule -DisplayName "Iriun Webcam (Inbound)" `
                        -Direction Inbound `
                        -Action Allow `
                        -Program $iriunPath `
                        -Profile Any `
                        -Description "Allow Iriun Webcam to receive video from phone"
    
    Write-Host "✔ Firewall rules added successfully." -ForegroundColor Green
} else {
    Write-Error "Iriun Webcam not found at $iriunPath. Please install it first from iriun.com"
}

# 3. Enable Network Discovery (SSDP and UPnP)
Write-Host "[2/3] Enabling Network Discovery services..." -ForegroundColor Yellow
Set-Service -Name "SSDPSRV" -StartupType Automatic
Start-Service -Name "SSDPSRV" -ErrorAction SilentlyContinue
Set-Service -Name "upnphost" -StartupType Automatic
Start-Service -Name "upnphost" -ErrorAction SilentlyContinue
Write-Host "✔ Network discovery services started." -ForegroundColor Green

# 4. Set Current Wi-Fi Network to Private (Windows blocks discovery on 'Public')
Write-Host "[3/3] Optimizing network profile..." -ForegroundColor Yellow
$network = Get-NetConnectionProfile | Where-Object { $_.IPv4Connectivity -eq 'Internet' -or $_.InterfaceAlias -like "*Wi-Fi*" }
if ($network) {
    if ($network.NetworkCategory -ne 'Private') {
        Set-NetConnectionProfile -InterfaceIndex $network.InterfaceIndex -NetworkCategory Private
        Write-Host "✔ Network set to 'Private' (Required for device discovery)." -ForegroundColor Green
    } else {
        Write-Host "✔ Network is already set to Private." -ForegroundColor Green
    }
}

Write-Host "`n--- FIX COMPLETE ---" -ForegroundColor Cyan
Write-Host "1. RESTART the Iriun Desktop app on your PC."
Write-Host "2. OPEN the Iriun app on your phone."
Write-Host "3. They should now connect automatically over Wi-Fi."
Read-Host "Press Enter to exit..."
