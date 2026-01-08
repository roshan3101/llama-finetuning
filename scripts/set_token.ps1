# PowerShell script to set Hugging Face token
# Usage: .\scripts\set_token.ps1

Write-Host "Setting Hugging Face Token" -ForegroundColor Cyan
Write-Host ""

# Check if .env file exists
$envFile = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $envFile) {
    Write-Host "Found .env file at: $envFile" -ForegroundColor Green
} else {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    New-Item -Path $envFile -ItemType File -Force | Out-Null
}

# Get token from user
$token = Read-Host "Enter your Hugging Face token (from https://huggingface.co/settings/tokens)"

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "No token provided. Exiting." -ForegroundColor Red
    exit 1
}

# Validate token format
if (-not $token.StartsWith("hf_") -and $token.Length -lt 20) {
    Write-Host "Warning: Token format may be incorrect. HF tokens usually start with 'hf_' and are ~37 characters." -ForegroundColor Yellow
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

# Write to .env file
$content = "HF_TOKEN=$token"
Set-Content -Path $envFile -Value $content -Force

Write-Host ""
Write-Host "Token saved to .env file!" -ForegroundColor Green
Write-Host ""

# Also set as environment variable for current session
$env:HF_TOKEN = $token
Write-Host "Token also set for current PowerShell session." -ForegroundColor Green
Write-Host ""
Write-Host "To make it permanent, restart your terminal or run:" -ForegroundColor Cyan
Write-Host '  $env:HF_TOKEN="' + $token + '"' -ForegroundColor Yellow
Write-Host ""

