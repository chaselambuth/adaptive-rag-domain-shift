param([string]$ConfigPath)

$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $ConfigPath) {
    $candidates = @((Join-Path $project 'config.json'),
                    (Join-Path (Split-Path $project -Parent) 'config.json'))
    $ConfigPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $ConfigPath -or -not (Test-Path -LiteralPath $ConfigPath)) {
    throw 'No local config.json found. Pass -ConfigPath with the path to your private configuration file.'
}

$settings = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$apiKey = if ($settings.OPENAI_API_KEY) { $settings.OPENAI_API_KEY } else { $settings.API_KEY }
if (-not $apiKey) {
    throw 'config.json must contain OPENAI_API_KEY or API_KEY.'
}

# Docker Compose reads the key from this process; nothing is copied into the Git repository.
$previous = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'Process')
try {
    [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', $apiKey, 'Process')
    Push-Location $project
    try {
        docker compose up -d --build flask
        if ($LASTEXITCODE -ne 0) { throw 'Docker Compose could not start the Flask app.' }
    } finally {
        Pop-Location
    }
} finally {
    [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', $previous, 'Process')
}
