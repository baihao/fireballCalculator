function Write-Status {
    param([string]$Level, [string]$Message)
    Write-Host "[$Level] $Message"
}
Write-Status OK "test"
