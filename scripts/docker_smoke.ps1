param(
    [string]$PrivateRoot = "$(Split-Path -Parent $PSScriptRoot)\..\support-ticket-triage-ml-private"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedPrivate = (Resolve-Path -LiteralPath $PrivateRoot).Path
$modelConfig = Join-Path $projectRoot "configs\evaluation.json"

if (-not (Test-Path -LiteralPath $modelConfig)) {
    throw "Frozen evaluation config is missing. Run validation freeze first."
}

$image = "support-ticket-triage-ml:local-smoke"
$container = "support-ticket-triage-smoke"
docker build --tag $image $projectRoot
try {
    docker run --detach --rm --name $container --publish 8000:8000 --volume "${resolvedPrivate}:/artifacts:ro" $image | Out-Null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 2
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 3
            if ($health.status -eq "ready") {
                Write-Output "Docker smoke test passed."
                exit 0
            }
        } catch {
            if ($attempt -eq 29) { throw }
        }
    }
    throw "Container did not become healthy."
} finally {
    docker stop $container 2>$null | Out-Null
}
