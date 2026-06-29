# watch.ps1 — auto-commit and push any change in the invert directory
# Usage: right-click → "Run with PowerShell"  OR  pwsh -File watch.ps1

$repoPath = $PSScriptRoot

Write-Host "Watching $repoPath for changes..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop.`n" -ForegroundColor DarkGray

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path                  = $repoPath
$watcher.Filter                = "*.*"
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter          = [System.IO.NotifyFilters]::LastWrite `
                               -bor [System.IO.NotifyFilters]::FileName

# Debounce: only push once per burst of rapid saves
$debounceTimer = $null
$debounceMs    = 2000   # wait 2 s after last change before pushing

$pushAction = {
    param($path)
    $skip = @('.git', 'watch.ps1')
    foreach ($s in $skip) { if ($path -like "*$s*") { return } }

    Write-Host "[$(Get-Date -f 'HH:mm:ss')] Change detected: $path" -ForegroundColor Yellow

    Push-Location $repoPath
    git add -A
    $status = git status --porcelain
    if ($status) {
        $msg = "Auto-update: $(Split-Path $path -Leaf)"
        git commit -m $msg
        git push origin main
        Write-Host "  Pushed.`n" -ForegroundColor Green
    } else {
        Write-Host "  Nothing to commit.`n" -ForegroundColor DarkGray
    }
    Pop-Location
}

$onChange = {
    $p = $Event.SourceEventArgs.FullPath
    # Reset debounce timer on each event
    if ($null -ne $script:debounceTimer) { $script:debounceTimer.Stop() }
    $script:debounceTimer = [System.Timers.Timer]::new($using:debounceMs)
    $script:debounceTimer.AutoReset = $false
    Register-ObjectEvent $script:debounceTimer Elapsed -Action {
        & $using:pushAction $using:p
        $script:debounceTimer.Dispose()
    } | Out-Null
    $script:debounceTimer.Start()
}

Register-ObjectEvent $watcher Changed -Action $onChange | Out-Null
Register-ObjectEvent $watcher Created -Action $onChange | Out-Null
Register-ObjectEvent $watcher Renamed -Action $onChange | Out-Null

$watcher.EnableRaisingEvents = $true

# Keep the script alive
try {
    while ($true) { Start-Sleep -Seconds 5 }
} finally {
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "Watcher stopped." -ForegroundColor DarkGray
}
