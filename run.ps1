$ErrorActionPreference = "Stop"

$AppModule = "mlstudio/main.py"
$SyncLog = [System.IO.Path]::GetTempFileName()

function Test-NetworkError {
    param([string]$Path)

    $Pattern = "network|internet|connection|connect|timed out|timeout|dns|temporary failure|name resolution|failed to resolve|could not resolve|proxy|ssl|tls|certificate|offline"
    return (Select-String -Path $Path -Pattern $Pattern -CaseSensitive:$false -Quiet)
}

Write-Host "Running uv sync..."
& uv sync *> $SyncLog
$SyncStatus = $LASTEXITCODE

if ($SyncStatus -eq 0) {
    Get-Content $SyncLog
}
elseif (Test-NetworkError -Path $SyncLog) {
    Write-Warning "uv sync failed because the network appears unavailable. Continuing with the existing environment."
    Get-Content $SyncLog
}
else {
    Write-Error "uv sync failed with a non-network error:" -ErrorAction Continue
    Get-Content $SyncLog | ForEach-Object { Write-Error $_ -ErrorAction Continue }
    Remove-Item $SyncLog -ErrorAction SilentlyContinue
    exit $SyncStatus
}

Remove-Item $SyncLog -ErrorAction SilentlyContinue

Write-Host "Starting Streamlit..."
& uv run -m streamlit run $AppModule
$RunStatus = $LASTEXITCODE

if ($RunStatus -ne 0) {
    Write-Error "Streamlit failed to run. Check the error output above." -ErrorAction Continue
    exit $RunStatus
}
